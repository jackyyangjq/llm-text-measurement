import json

from llmtm.schema import check, instructions, parse_batch

TEXT = "Staff were super friendly. The room was small and the wifi did not work."


def test_evidence_must_be_in_the_text():
    answer = {
        "overall": "mixed",
        "confidence": 0.9,
        "comments": [
            {"aspect": "staff", "polarity": "positive", "evidence": "Staff were super friendly"},
            {"aspect": "room", "polarity": "negative", "evidence": "the room was  SMALL"},
            {"aspect": "facilities", "polarity": "negative", "evidence": "internet was broken"},
            {"aspect": "spa", "polarity": "negative", "evidence": "the wifi did not work"},
        ],
    }
    c = check(answer, TEXT, "r1")
    assert [x["aspect"] for x in c.comments] == ["staff", "room"]
    assert [d["reason"] for d in c.dropped] == ["evidence not in text", "schema"]
    assert c.needs_review


def test_evidence_must_match_whole_words():
    c = check(
        {
            "overall": "positive",
            "confidence": 1,
            "comments": [{"aspect": "room", "polarity": "negative", "evidence": "oom was sma"}],
        },
        TEXT,
        "r1",
    )
    assert not c.comments


def test_bad_overall_and_confidence_are_problems():
    c = check({"overall": "great", "confidence": 3}, TEXT, "r1")
    assert len(c.problems) == 2 and c.overall is None


def test_parse_batch_reports_missing_ids_and_tolerates_prose_around_json():
    reply = "Here you go:\n" + json.dumps(
        {"reviews": [{"id": "a", "overall": "positive", "confidence": 0.8, "comments": []}]}
    )
    checked, missing = parse_batch(reply, {"a": TEXT, "b": TEXT})
    assert list(checked) == ["a"] and missing == ["b"]


def test_instructions_list_every_aspect():
    text = instructions()
    for aspect in ("location", "room", "cleanliness", "staff", "food", "value", "noise", "facilities"):
        assert f'"{aspect}"' in text
