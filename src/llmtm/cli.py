"""llmtm annotate | evaluate

export LLM_API_KEY=...            # any OpenAI-compatible endpoint; LLM_BASE_URL to change it
llmtm annotate --model gpt-5.5 --run main
llmtm annotate --model gpt-5.5 --run rerun --n 200 --seed 1 --no-second-pass
llmtm annotate --model claude-sonnet-5 --run cross --n 300 --seed 2 --no-second-pass
llmtm annotate --model gpt-5.5 --run controls --controls --no-second-pass
llmtm evaluate
"""

from __future__ import annotations

import argparse
from pathlib import Path

RESULTS = Path("results")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="llmtm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("annotate")
    a.add_argument("--model", required=True)
    a.add_argument("--run", required=True, help="name of the run; output goes to results/annotations/<run>.jsonl")
    a.add_argument("--n", type=int, help="annotate a random subset of this size")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--controls", action="store_true", help="annotate the negative-control texts instead")
    a.add_argument("--second-pass-below", type=float, default=0.8)
    a.add_argument("--no-second-pass", action="store_true")
    a.add_argument("--workers", type=int, default=4)
    sub.add_parser("evaluate", help="write results/evaluation.json and the figures")
    args = parser.parse_args(argv)

    if args.command == "annotate":
        from . import annotate, data
        from .client import ChatClient

        reviews = annotate.negative_controls() if args.controls else data.load_sample()
        if args.n:
            reviews = reviews.sample(args.n, random_state=args.seed)
        client = ChatClient(args.model, RESULTS / "usage.jsonl")
        threshold = None if args.no_second_pass else args.second_pass_below
        recs = annotate.run(
            reviews,
            client,
            RESULTS / "annotations" / f"{args.run}.jsonl",
            workers=args.workers,
            second_pass_below=threshold,
        )
        print(f"{args.run}: {len(recs)} reviews annotated with {args.model}")
    elif args.command == "evaluate":
        from .evaluate import write_all
        from .plots import make_figures

        write_all(RESULTS)
        make_figures(RESULTS, Path("figures"))


if __name__ == "__main__":
    main()
