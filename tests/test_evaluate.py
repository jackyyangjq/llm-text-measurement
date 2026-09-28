import pandas as pd
import pytest

from llmtm.annotate import EXAMPLES, Retriever, negative_controls
from llmtm.evaluate import compare_runs, mentions, polarity_labels, reviewer_truth


def rec(comments):
    return {
        "comments": [{"aspect": a, "polarity": p, "evidence": e} for a, p, e in comments],
        "overall": "mixed",
        "dropped": [],
        "problems": [],
        "route": "batch",
        "confidence": 0.9,
    }


A = {
    "r1": rec([("staff", "positive", "friendly staff"), ("room", "negative", "tiny room")]),
    "r2": rec([("noise", "negative", "loud street")]),
}
B = {
    "r1": rec([("staff", "positive", "friendly staff"), ("room", "positive", "tiny room")]),
    "r2": rec([("noise", "negative", "loud street"), ("value", "negative", "overpriced")]),
}


def test_mentions_and_polarity_labels():
    m = mentions(A, ["r1", "r2"])
    assert m.loc["r1", "staff"] == 1 and m.loc["r2", "staff"] == 0
    p = polarity_labels({"r1": rec([("room", "positive", "x"), ("room", "negative", "y")])}, ["r1"])
    assert p.loc["r1", "room"] == "both" and p.loc["r1", "food"] == "none"


def test_identical_runs_agree_perfectly_and_different_runs_do_not():
    same = compare_runs(A, A)
    assert same["alpha_mentions"] == 1 and same["identical_label_sets"] == 1
    diff = compare_runs(A, B)
    assert diff["alpha_mentions"] < 1 and diff["identical_label_sets"] == 0


def test_reviewer_truth_uses_the_part_the_evidence_comes_from():
    sample = pd.DataFrame({"review_id": ["r1"], "liked": ["friendly staff"], "disliked": ["tiny room"]})
    truth = reviewer_truth(
        {"r1": rec([("staff", "positive", "friendly staff"), ("room", "positive", "tiny room")])}, sample
    )
    assert truth.set_index("aspect")["reviewer"].to_dict() == {"staff": "positive", "room": "negative"}


def test_controls_and_retrieval():
    ctrl = negative_controls(10)
    assert len(ctrl) == 10 and ctrl["review_id"].is_unique
    nearest = Retriever().nearest("the swimming pool was closed", k=1)[0]
    assert "pool" in EXAMPLES[nearest][0]
    assert pytest.approx(1) == 1
