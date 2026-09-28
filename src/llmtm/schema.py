"""What the model is asked for, and the checks every answer has to pass.

The model reads a batch of reviews and returns, for each, the aspects the guest comments on, the
polarity of each comment and a verbatim evidence span. `check` rejects anything outside the
schema, and drops a comment whose evidence cannot be found word for word in the review: a
label the text does not support is treated as a hallucination, not as data.
"""

# ruff: noqa: E501  (the prompt text is kept exactly as it was sent)
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

PROMPT_VERSION = "v1"

ASPECTS = {
    "location": "where the hotel is: distance to sights, transport links, the neighbourhood",
    "room": "the room itself: size, bed, comfort, view, furniture, bathroom, temperature",
    "cleanliness": "how clean the room, bathroom or hotel was",
    "staff": "staff and service: reception, check-in, helpfulness, friendliness",
    "food": "breakfast, restaurant, bar, drinks",
    "value": "price, value for money, extra charges",
    "noise": "noise or quiet, including soundproofing",
    "facilities": "hotel facilities outside the room: wifi, lift, parking, pool, gym, spa, lobby",
}
POLARITIES = ("positive", "negative")
OVERALL = ("positive", "mixed", "negative")

INSTRUCTIONS = """You label hotel reviews for a research dataset.

For each review, list every comment the guest makes on these aspects:
{aspects}

For each comment return:
- "aspect": one of the names above
- "polarity": "positive" or "negative"
- "evidence": a span of 2 to 12 words copied from the review exactly as written (same words, same spelling, no paraphrase) that shows the comment

If the guest praises and criticises the same aspect, return two comments. Return nothing for aspects the review does not mention; do not infer them.

For each review also return "overall" ("positive", "mixed" or "negative") and "confidence", a number from 0 to 1 for how sure you are that the comments you listed are complete and correct.

Answer with one JSON object and nothing else:
{{"reviews": [{{"id": "...", "overall": "...", "confidence": 0.0, "comments": [{{"aspect": "...", "polarity": "...", "evidence": "..."}}]}}]}}
with one entry per review, using the ids given in brackets."""


def instructions() -> str:
    lines = "\n".join(f'- "{name}": {desc}' for name, desc in ASPECTS.items())
    return INSTRUCTIONS.format(aspects=lines)


def batch_message(reviews: list[tuple[str, str]]) -> str:
    return "Reviews:\n\n" + "\n\n".join(f"[{rid}] {text}" for rid, text in reviews)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass
class Checked:
    id: str
    overall: str | None
    confidence: float | None
    comments: list[dict] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)  # comments failing the schema or the evidence check
    problems: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.problems or self.dropped)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "overall": self.overall,
            "confidence": self.confidence,
            "comments": self.comments,
            "dropped": self.dropped,
            "problems": self.problems,
        }


def check(answer: dict, text: str, review_id: str) -> Checked:
    """Validate one review's answer against the schema and the review text."""
    out = Checked(review_id, None, None)
    overall = answer.get("overall")
    if overall in OVERALL:
        out.overall = overall
    else:
        out.problems.append(f"overall={overall!r}")
    try:
        conf = float(answer.get("confidence"))
        out.confidence = conf if 0 <= conf <= 1 else None
    except (TypeError, ValueError):
        pass
    if out.confidence is None:
        out.problems.append("confidence missing or out of range")
    haystack = f" {normalize(text)} "
    for c in answer.get("comments") or []:
        if not isinstance(c, dict) or c.get("aspect") not in ASPECTS or c.get("polarity") not in POLARITIES:
            out.dropped.append({"comment": c, "reason": "schema"})
        elif not normalize(str(c.get("evidence", ""))) or f" {normalize(c['evidence'])} " not in haystack:
            out.dropped.append({"comment": c, "reason": "evidence not in text"})
        else:
            out.comments.append({"aspect": c["aspect"], "polarity": c["polarity"], "evidence": c["evidence"]})
    return out


def parse_batch(content: str, reviews: dict[str, str]) -> tuple[dict[str, Checked], list[str]]:
    """Checked answers by review id, and the ids the reply left out."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.S)
        payload = json.loads(match.group(0)) if match else {}
    answers = {str(a.get("id")): a for a in payload.get("reviews", []) if isinstance(a, dict)}
    checked = {rid: check(answers[rid], text, rid) for rid, text in reviews.items() if rid in answers}
    return checked, [rid for rid in reviews if rid not in answers]
