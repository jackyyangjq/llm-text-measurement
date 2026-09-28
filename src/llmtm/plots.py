"""The README figures, from results/ (no model calls)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import annotate, data  # noqa: E402
from .evaluate import compare_runs, polarity_labels  # noqa: E402
from .schema import ASPECTS  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f4f8fe", "#9ec5f4", "#2a78d6", "#0d366b"])
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
    }
)


def validation_figure(results: Path, path: Path) -> None:
    ann = results / "annotations"
    main = annotate.load(ann / "main.jsonl")
    ev = json.loads((results / "evaluation.json").read_text())
    subs = pd.read_csv(results / "substitutes.csv").pivot(index="aspect", columns="method", values="f1")
    series = {}
    for run, label in (("rerun", "Same model, second run (alpha)"), ("cross", "Other provider's model (alpha)")):
        if (ann / f"{run}.jsonl").exists():
            series[label] = pd.Series(compare_runs(main, annotate.load(ann / f"{run}.jsonl"))["per_aspect_alpha"])
    series["TF-IDF student vs LLM (F1)"] = subs["student"]
    series["Word2Vec dictionary vs LLM (F1)"] = subs["dictionary"]
    order = list(ASPECTS)[::-1]
    per = ev["reviewer_truth"]["per_aspect"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.5, 1]}, sharey=True)
    y = np.arange(len(order))
    for (label, s), color, marker in zip(series.items(), [BLUE, ORANGE, AQUA, YELLOW], "osDv", strict=False):
        a1.scatter(
            s.reindex(order),
            y,
            s=55,
            color=color,
            marker=marker,
            edgecolor=SURFACE,
            linewidth=1.5,
            label=label,
            zorder=3,
        )
    a1.set_yticks(y, order)
    a1.set_xlim(0, 1.02)
    a1.set_xlabel("agreement on whether the review mentions the aspect")
    a1.set_title("Which aspects a review mentions")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8.5)
    acc = pd.Series({k: v["mean"] for k, v in per.items()}).reindex(order)
    n = pd.Series({k: v["size"] for k, v in per.items()}).reindex(order)
    a2.barh(y, acc, color=BLUE, height=0.55)
    for yi, (a, k) in enumerate(zip(acc, n, strict=True)):
        a2.text(min(a, 1) - 0.02, yi, f"{a:.0%}  (n={k})", va="center", ha="right", color="white", fontsize=8.5)
    a2.set_xlim(0, 1)
    a2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    a2.grid(axis="y", visible=False)
    a2.set_title("Polarity vs the guest's own liked/disliked split")
    a2.set_xlabel("share of comments where the model agrees")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def score_figure(results: Path, path: Path) -> None:
    main = annotate.load(results / "annotations" / "main.jsonl")
    sample = data.load_sample().set_index("review_id").loc[list(main)]
    pol = polarity_labels(main, list(main))
    negative = pol.isin(["negative", "both"]).astype(float)
    bands = pd.cut(sample["score"], [0, 5, 7, 8.5, 10], labels=["up to 5", "5 to 7", "7 to 8.5", "8.5 to 10"])
    order = list(bands.cat.categories)
    table = negative.groupby(bands.astype(str).to_numpy()).mean().T[order]
    counts = bands.value_counts().reindex(order)
    table = table.loc[table.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.imshow(table.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=0.6, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color="white" if v > 0.35 else INK)
    ax.set_xticks(range(table.shape[1]), [f"{c}\n(n={k})" for c, k in counts.items()])
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_xlabel("guest's score")
    ax.set_title("Share of reviews with a negative comment on each aspect")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def make_figures(results: Path, figures: Path) -> None:
    figures.mkdir(exist_ok=True)
    validation_figure(results, figures / "validation.png")
    score_figure(results, figures / "aspects_by_score.png")
