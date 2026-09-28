"""Build data/reviews_sample.csv and results/dictionary.json from the full review file.

The full file (238 MB) is downloaded to data/raw/, used, and can be deleted afterwards: nothing
else in the repository reads it.

    python scripts/prepare_data.py            # downloads if data/raw/Hotel_Reviews.csv is missing
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llmtm import data, dictionary  # noqa: E402

RAW = Path("data/raw/Hotel_Reviews.csv")


def main() -> None:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(data.RAW_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with RAW.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    sample = data.prepare_sample(RAW)
    sample.to_csv(data.SAMPLE, index=False)
    print(f"sample: {len(sample)} reviews -> {data.SAMPLE}", sample["kind"].value_counts().to_dict())

    full = pd.read_csv(RAW, usecols=["Positive_Review", "Negative_Review"]).drop(index=sample["row"])
    texts = (full["Positive_Review"] + " " + full["Negative_Review"]).tolist()
    start = time.perf_counter()
    model = dictionary.train_embeddings(texts)
    expanded = dictionary.expand(model)
    dictionary.save(expanded, Path("results/dictionary.json"))
    print(
        f"Word2Vec on {len(texts):,} reviews in {time.perf_counter() - start:.0f}s; dictionary sizes:",
        {k: len(v) for k, v in expanded.items()},
    )


if __name__ == "__main__":
    main()
