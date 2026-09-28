"""The review sample: public hotel reviews whose authors split them into what they liked and disliked.

Source: "515K Hotel Reviews Data in Europe" (Booking.com, 2015-2017), released on Kaggle under CC0
by Jiashen Liu. Each review has a `Positive_Review` and a `Negative_Review` field written by the
guest. We join the two fields, in random order, into one text for the model; the fields
themselves then serve as the reviewer's own label for every piece of text, which lets us check
the polarity the model assigns without hand-coding.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_URL = "https://huggingface.co/datasets/Dricz/515k-Hotel-Reviews-In-Europe/resolve/main/Hotel_Reviews.csv"
SAMPLE = Path("data/reviews_sample.csv")
# Placeholder texts guests leave in a field they have nothing to say in
EMPTY = re.compile(r"^(no (negative|positive)|nothing.*|none|n ?a|na|nil|all good|everything)$")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def usable(text: str, min_words: int) -> bool:
    t = clean(text).lower()
    return not EMPTY.match(t) and len(t.split()) >= min_words


def prepare_sample(
    raw: Path, n_mixed: int = 700, n_positive: int = 200, n_negative: int = 100, seed: int = 0
) -> pd.DataFrame:
    """Draw the sample: reviews with both a liked and a disliked part (the informative case), and
    reviews with only one; texts of 8 to 150 words; stratified by year of stay.

    `text` joins the two parts in random order with a full stop; `first` records which came first;
    `row` is the review's position in the source file.
    """
    df = pd.read_csv(raw)
    df["row"] = df.index  # position in the source file, so the sample can be held out elsewhere
    df["date"] = pd.to_datetime(df["Review_Date"], format="%m/%d/%Y")
    df["pos"] = df["Positive_Review"].map(clean)
    df["neg"] = df["Negative_Review"].map(clean)
    has_pos = df["pos"].map(lambda t: usable(t, 3))
    has_neg = df["neg"].map(lambda t: usable(t, 3))
    words = df["pos"].str.split().str.len() + df["neg"].str.split().str.len()
    df = df[(words >= 8) & (words <= 150)]
    rng = np.random.default_rng(seed)
    groups = {
        "mixed": df[has_pos & has_neg],
        "positive only": df[has_pos & ~has_neg],
        "negative only": df[~has_pos & has_neg],
    }
    picked = []
    for kind, n in (("mixed", n_mixed), ("positive only", n_positive), ("negative only", n_negative)):
        g = groups[kind]
        per_year = g.groupby(g["date"].dt.year, group_keys=False).apply(
            lambda x, k=n, g=g: x.sample(min(len(x), round(k * len(x) / len(g))), random_state=int(rng.integers(2**31)))
        )
        picked.append(per_year.assign(kind=kind))
    s = pd.concat(picked).reset_index(drop=True)
    s.loc[s["kind"] == "positive only", "neg"] = ""
    s.loc[s["kind"] == "negative only", "pos"] = ""
    first_pos = rng.random(len(s)) < 0.5
    parts = [(p, n) if fp else (n, p) for p, n, fp in zip(s["pos"], s["neg"], first_pos, strict=True)]
    s["text"] = [". ".join(x for x in pair if x) + "." for pair in parts]
    s["first"] = np.where(first_pos, "positive", "negative")
    s["review_id"] = [f"r{i:04d}" for i in range(len(s))]
    s["country"] = s["Hotel_Address"].str.split().str[-1].replace({"Kingdom": "United Kingdom"})
    out = s[["review_id", "row", "date", "country", "Reviewer_Score", "kind", "first", "pos", "neg", "text"]]
    return out.rename(columns={"Reviewer_Score": "score", "pos": "liked", "neg": "disliked"})


def load_sample(path: Path = SAMPLE) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, parse_dates=["date"])
    return df


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text).lower())
