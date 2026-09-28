"""Batch annotation with a cache, a second pass for doubtful answers, and negative controls.

- Reviews go to the model ten at a time. Every checked answer is appended to a JSONL file per
  run, so a run that stops half-way resumes where it stopped and nothing is paid for twice.
- A review goes to a second pass when its answer broke the schema, cited evidence that is not in
  the text, or came with confidence below `second_pass_below`. The second pass sends it alone,
  with the three most similar worked examples from a small hand-labelled bank (retrieved by
  TF-IDF similarity) as few-shot demonstrations.
- `negative_controls` builds texts that mention no hotel aspect at all; every comment the model
  returns for them is a false positive.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .client import ChatClient
from .schema import PROMPT_VERSION, batch_message, check, instructions, parse_batch

EXAMPLES = [
    (
        "The room was tiny and the shower barely worked. Staff at reception were lovely and upgraded us the next day.",
        [
            ("room", "negative", "The room was tiny"),
            ("room", "negative", "the shower barely worked"),
            ("staff", "positive", "Staff at reception were lovely"),
        ],
        "mixed",
    ),
    (
        "Nothing to complain about. Great location right next to the station and a very good breakfast.",
        [
            ("location", "positive", "Great location right next to the station"),
            ("food", "positive", "a very good breakfast"),
        ],
        "positive",
    ),
    (
        "For 300 euros a night I expected better. The street outside was loud until 3am.",
        [
            ("value", "negative", "For 300 euros a night I expected better"),
            ("noise", "negative", "The street outside was loud"),
        ],
        "negative",
    ),
    (
        "Breakfast was expensive at 25 each but the choice was excellent.",
        [("value", "negative", "Breakfast was expensive at 25 each"), ("food", "positive", "the choice was excellent")],
        "mixed",
    ),
    (
        "Carpet in the corridor was stained and our bathroom had hair in the sink. Bed was comfortable though.",
        [
            ("cleanliness", "negative", "Carpet in the corridor was stained"),
            ("cleanliness", "negative", "our bathroom had hair in the sink"),
            ("room", "positive", "Bed was comfortable"),
        ],
        "mixed",
    ),
    (
        "Wifi kept dropping and there is only one small lift for eight floors.",
        [("facilities", "negative", "Wifi kept dropping"), ("facilities", "negative", "only one small lift")],
        "negative",
    ),
    (
        "We were in town for a wedding. Everything was perfect, will come back.",
        [],
        "positive",
    ),
    (
        "Quiet room facing the courtyard, slept really well. Check in took forty minutes.",
        [
            ("noise", "positive", "Quiet room facing the courtyard"),
            ("staff", "negative", "Check in took forty minutes"),
        ],
        "mixed",
    ),
    (
        "Good value for London. Ten minutes walk to Hyde Park.",
        [("value", "positive", "Good value for London"), ("location", "positive", "Ten minutes walk to Hyde Park")],
        "positive",
    ),
    (
        "The pool was closed for the whole stay and nobody told us when we booked.",
        [("facilities", "negative", "The pool was closed for the whole stay")],
        "negative",
    ),
    (
        "Clean modern rooms, but the air conditioning was noisy all night.",
        [
            ("cleanliness", "positive", "Clean modern rooms"),
            ("room", "positive", "Clean modern rooms"),
            ("noise", "negative", "the air conditioning was noisy all night"),
        ],
        "mixed",
    ),
    (
        "The concierge Marco recommended a fantastic restaurant nearby and booked it for us.",
        [("staff", "positive", "recommended a fantastic restaurant nearby and booked it for us")],
        "positive",
    ),
]


def _example_block(indices: list[int]) -> str:
    blocks = []
    for n, i in enumerate(indices, 1):
        text, comments, overall = EXAMPLES[i]
        answer = {
            "id": f"x{n}",
            "overall": overall,
            "confidence": 0.95,
            "comments": [{"aspect": a, "polarity": p, "evidence": e} for a, p, e in comments],
        }
        blocks.append(f"[x{n}] {text}\nAnswer: {json.dumps({'reviews': [answer]})}")
    return "Worked examples:\n\n" + "\n\n".join(blocks)


class Retriever:
    """TF-IDF nearest neighbours in the example bank."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([t for t, _, _ in EXAMPLES])

    def nearest(self, text: str, k: int = 3) -> list[int]:
        sims = (self.matrix @ self.vectorizer.transform([text]).T).toarray().ravel()
        return list(sims.argsort()[::-1][:k])


def load(path: Path) -> dict[str, dict]:
    """Last record per review id (a second pass supersedes the first)."""
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def run(
    reviews: pd.DataFrame,
    client: ChatClient,
    out: Path,
    batch_size: int = 10,
    workers: int = 4,
    second_pass_below: float | None = 0.6,
) -> dict[str, dict]:
    """Annotate `reviews` (columns review_id, text) into `out`; returns the records by id."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load(out)
    todo = reviews[~reviews["review_id"].isin(done)]
    texts = dict(zip(reviews["review_id"], reviews["text"], strict=True))
    batches = [todo.iloc[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    system = instructions()

    def write(records: list[dict]) -> None:
        with out.open("a") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def one_batch(batch: pd.DataFrame) -> tuple[list[dict], list[str]]:
        items = dict(zip(batch["review_id"], batch["text"], strict=True))
        reply = client.chat(system, batch_message(list(items.items())), tag=f"{out.stem}:{batch['review_id'].iloc[0]}")
        checked, missing = parse_batch(reply.content, items)
        recs = [
            {**c.to_json(), "model": reply.model, "prompt": PROMPT_VERSION, "route": "batch"} for c in checked.values()
        ]
        return recs, missing

    leftovers: list[str] = []
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one_batch, b) for b in batches]
        for n, fut in enumerate(as_completed(futures), 1):
            recs, missing = fut.result()
            write(recs)
            leftovers += missing
            if n % 10 == 0 or n == len(futures):
                print(f"  {out.stem}: {n}/{len(futures)} batches", flush=True)

    records = load(out)
    if second_pass_below is None:
        doubtful = list(leftovers)
    else:
        doubtful = leftovers + [
            rid
            for rid, rec in records.items()
            if rec["route"] == "batch"
            and (rec["problems"] or rec["dropped"] or (rec["confidence"] or 0) < second_pass_below)
        ]
    if doubtful:
        retriever = Retriever()

        def single(rid: str) -> dict:
            text = texts[rid]
            shots = _example_block(retriever.nearest(text))
            reply = client.chat(system, shots + "\n\n" + batch_message([(rid, text)]), tag=f"{out.stem}:2nd:{rid}")
            checked, _ = parse_batch(reply.content, {rid: text})
            c = checked.get(rid) or check({}, text, rid)
            return {**c.to_json(), "model": reply.model, "prompt": PROMPT_VERSION, "route": "second pass"}

        with ThreadPoolExecutor(workers) as pool:
            write(list(pool.map(single, sorted(set(doubtful)))))
        print(f"  {out.stem}: second pass on {len(set(doubtful))} reviews", flush=True)
    return load(out)


def negative_controls(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Short travel narratives that say nothing about the hotel."""
    rng = random.Random(seed)
    cities = ["Vienna", "Paris", "London", "Milan", "Barcelona", "Amsterdam"]
    parts = [
        "We came to {city} for {reason}.",
        "Our flight from {origin} landed at {hour} in the morning.",
        "It was our {nth} visit to {city}.",
        "We stayed {k} nights in {month}.",
        "My {relative} booked the trip as a birthday present.",
        "The museum we wanted to see was closed on Monday.",
        "It rained for most of the week, so we spent a lot of time in cafes.",
        "We took the train to {city2} for a day trip.",
        "The conference ended on Thursday afternoon.",
        "We used the city bike scheme to get around.",
    ]
    fill = {
        "reason": ["a conference", "a wedding", "a football match", "a school trip", "work"],
        "origin": ["Toronto", "Dubai", "Sydney", "New York", "Singapore"],
        "hour": ["six", "seven", "nine", "eleven"],
        "nth": ["first", "second", "third"],
        "k": ["two", "three", "four", "five"],
        "month": ["March", "July", "October", "December"],
        "relative": ["sister", "husband", "daughter", "friend"],
    }
    rows = []
    for i in range(n):
        city, city2 = rng.sample(cities, 2)
        sentences = rng.sample(parts, 3)
        values = {k: rng.choice(v) for k, v in fill.items()}
        text = " ".join(s.format(city=city, city2=city2, **values) for s in sentences)
        rows.append({"review_id": f"c{i:03d}", "text": text})
    return pd.DataFrame(rows)
