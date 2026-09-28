"""Every check on the measurement, from the saved annotations (no model calls).

1. Grounding: how many comments cite evidence that is really in the review.
2. Reviewer ground truth: a comment whose evidence lies in the part the guest filed under
   "liked" should be positive, under "disliked" negative.
3. Overall label against the guest's 1-10 score.
4. Test-retest: the same model run twice on 200 reviews.
5. Cross-model: a second model from another provider on 300 reviews.
6. Negative controls: texts with no hotel aspect.
7. Cheaper substitutes: a TF-IDF classifier trained on the LLM's labels, and the Word2Vec
   dictionary, both scored against the LLM on held-out reviews.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import annotate, data, dictionary
from .agreement import bootstrap_ci, cohen_kappa, krippendorff_alpha_nominal, precision_recall_f1
from .client import read_ledger
from .schema import ASPECTS, normalize


def mentions(records: dict[str, dict], ids: list[str]) -> pd.DataFrame:
    """1 where the review has at least one comment on the aspect."""
    rows = {rid: {a: 0 for a in ASPECTS} for rid in ids}
    for rid in ids:
        for c in records[rid]["comments"]:
            rows[rid][c["aspect"]] = 1
    return pd.DataFrame.from_dict(rows, orient="index")[list(ASPECTS)]


def polarity_labels(records: dict[str, dict], ids: list[str]) -> pd.DataFrame:
    """Per review and aspect: 'positive', 'negative', 'both' or 'none'."""
    out = {}
    for rid in ids:
        seen = {a: set() for a in ASPECTS}
        for c in records[rid]["comments"]:
            seen[c["aspect"]].add(c["polarity"])
        out[rid] = {a: ("both" if len(s) == 2 else next(iter(s)) if s else "none") for a, s in seen.items()}
    return pd.DataFrame.from_dict(out, orient="index")[list(ASPECTS)]


def grounding(records: dict[str, dict]) -> dict:
    kept = sum(len(r["comments"]) for r in records.values())
    dropped = [d for r in records.values() for d in r["dropped"]]
    ungrounded = sum(d["reason"] == "evidence not in text" for d in dropped)
    return {
        "reviews": len(records),
        "comments_returned": kept + len(dropped),
        "comments_kept": kept,
        "evidence_not_in_text": ungrounded,
        "schema_errors": len(dropped) - ungrounded,
        "grounded_share": kept / (kept + len(dropped)) if kept + len(dropped) else np.nan,
        "second_pass": sum(r["route"] == "second pass" for r in records.values()),
    }


def reviewer_truth(records: dict[str, dict], sample: pd.DataFrame) -> pd.DataFrame:
    """One row per kept comment whose evidence sits wholly inside the liked or the disliked part."""
    by_id = sample.set_index("review_id")
    rows = []
    for rid, rec in records.items():
        liked = f" {normalize(by_id.loc[rid, 'liked'])} "
        disliked = f" {normalize(by_id.loc[rid, 'disliked'])} "
        for c in rec["comments"]:
            ev = f" {normalize(c['evidence'])} "
            in_l, in_d = ev in liked, ev in disliked
            if in_l != in_d:
                rows.append(
                    {
                        "review_id": rid,
                        "aspect": c["aspect"],
                        "model": c["polarity"],
                        "reviewer": "positive" if in_l else "negative",
                        "evidence": c["evidence"],
                    }
                )
    return pd.DataFrame(rows)


def compare_runs(a: dict[str, dict], b: dict[str, dict], seed: int = 0) -> dict:
    """Agreement between two annotations of the same reviews."""
    ids = sorted(set(a) & set(b))
    ma, mb = mentions(a, ids), mentions(b, ids)
    pa, pb = polarity_labels(a, ids), polarity_labels(b, ids)

    def alpha_mentions(idx: np.ndarray) -> float:
        x, y = ma.to_numpy()[idx].ravel(), mb.to_numpy()[idx].ravel()
        return krippendorff_alpha_nominal(list(zip(x, y, strict=True)))

    def alpha_polarity(idx: np.ndarray) -> float:
        x, y = pa.to_numpy()[idx].ravel(), pb.to_numpy()[idx].ravel()
        both = (x != "none") & (y != "none")
        return krippendorff_alpha_nominal(list(zip(x[both], y[both], strict=True)))

    idx = np.arange(len(ids))
    per_aspect = {asp: krippendorff_alpha_nominal(list(zip(ma[asp], mb[asp], strict=True))) for asp in ASPECTS}
    overall_a = [a[i]["overall"] for i in ids]
    overall_b = [b[i]["overall"] for i in ids]
    return {
        "reviews": len(ids),
        "alpha_mentions": alpha_mentions(idx),
        "alpha_mentions_ci": bootstrap_ci(alpha_mentions, len(ids), seed=seed),
        "alpha_polarity": alpha_polarity(idx),
        "alpha_polarity_ci": bootstrap_ci(alpha_polarity, len(ids), seed=seed),
        "alpha_overall": krippendorff_alpha_nominal(list(zip(overall_a, overall_b, strict=True))),
        "identical_label_sets": float((pa == pb).all(axis=1).mean()),
        "per_aspect_alpha": per_aspect,
    }


def substitutes(records: dict[str, dict], sample: pd.DataFrame, dict_path: Path, seed: int = 0) -> pd.DataFrame:
    """F1 against the LLM's mention labels on 30% held-out reviews, for a TF-IDF + logistic
    regression student trained on the other 70% and for the Word2Vec dictionary."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    ids = sample["review_id"].tolist()
    y = mentions(records, ids)
    texts = sample.set_index("review_id").loc[ids, "text"]
    rng = np.random.default_rng(seed)
    test = rng.random(len(ids)) < 0.3
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = vec.fit_transform(texts[~test])
    X_test = vec.transform(texts[test])
    tagged = dictionary.tag(texts[test], dictionary.load(dict_path))
    rows = []
    for asp in ASPECTS:
        clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000).fit(X_train, y[asp][~test])
        student = clf.predict(X_test)
        for name, pred in (("student", student), ("dictionary", tagged[asp].to_numpy())):
            p, r, f = precision_recall_f1(y[asp][test], pred)
            rows.append(
                {"aspect": asp, "method": name, "precision": p, "recall": r, "f1": f, "llm_share": y[asp][test].mean()}
            )
    return pd.DataFrame(rows)


def usage(results: Path) -> pd.DataFrame:
    ledger = pd.DataFrame(read_ledger(results / "usage.jsonl"))
    ledger["run"] = ledger["tag"].str.split(":").str[0]
    ledger["pass"] = np.where(ledger["tag"].str.contains(":2nd:"), "second pass", "batch")
    return ledger.groupby(["run", "model", "pass"]).agg(
        calls=("tag", "size"),
        prompt_tokens=("prompt_tokens", "sum"),
        completion_tokens=("completion_tokens", "sum"),
        median_latency_s=("latency_s", "median"),
    )


def write_all(results: Path) -> dict:
    sample = data.load_sample()
    ann = results / "annotations"
    main = annotate.load(ann / "main.jsonl")
    out: dict = {"grounding": grounding(main)}

    truth = reviewer_truth(main, sample)
    agree = truth["model"] == truth["reviewer"]
    out["reviewer_truth"] = {
        "comments": len(truth),
        "accuracy": float(agree.mean()),
        "accuracy_ci": bootstrap_ci(lambda i: float(agree.to_numpy()[i].mean()), len(truth)),
        "kappa": cohen_kappa(truth["model"], truth["reviewer"]),
        "per_aspect": truth.assign(agree=agree).groupby("aspect")["agree"].agg(["mean", "size"]).to_dict("index"),
    }
    truth[~agree].to_csv(results / "reviewer_disagreements.csv", index=False)

    s = sample.set_index("review_id").loc[list(main)]
    overall = pd.Series({rid: r["overall"] for rid, r in main.items()})
    code = overall.map({"negative": 0, "mixed": 1, "positive": 2})
    out["overall_vs_score"] = {
        "spearman": float(stats.spearmanr(code, s["score"]).statistic),
        "mean_score": s.groupby(overall.reindex(s.index))["score"].mean().round(2).to_dict(),
        "counts": overall.value_counts().to_dict(),
    }

    for run in ("rerun", "cross"):
        if (ann / f"{run}.jsonl").exists():
            out[run] = compare_runs(main, annotate.load(ann / f"{run}.jsonl"))
    if (ann / "controls.jsonl").exists():
        ctrl = annotate.load(ann / "controls.jsonl")
        n_comments = [len(r["comments"]) + len(r["dropped"]) for r in ctrl.values()]
        out["controls"] = {
            "texts": len(ctrl),
            "texts_with_any_comment": int(sum(n > 0 for n in n_comments)),
            "comments": int(sum(n_comments)),
        }

    subs = substitutes(main, sample, results / "dictionary.json")
    subs.to_csv(results / "substitutes.csv", index=False)
    out["substitutes_macro_f1"] = subs.groupby("method")["f1"].mean().to_dict()
    out["aspect_shares"] = mentions(main, list(main)).mean().round(3).to_dict()

    use = usage(results)
    use.to_csv(results / "usage_summary.csv")
    out["tokens"] = {
        "prompt": int(use["prompt_tokens"].sum()),
        "completion": int(use["completion_tokens"].sum()),
        "calls": int(use["calls"].sum()),
    }
    (results / "evaluation.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))
    return out
