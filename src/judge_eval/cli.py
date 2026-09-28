"""Entry point for the judge-eval CLI.

    judge-eval generate-dataset --n-samples 50
    judge-eval run-evaluation --judge custom --judge-model openai:/gpt-5.4-mini-2026-03-17 --split opt
"""
import argparse

from judge_eval.common.models import DEFAULT_JUDGE_MODEL
from judge_eval.common.tracking import enable_autologging
from judge_eval.dataset.generate import main as generate_dataset
from judge_eval.harness.run import JUDGE_BACKENDS
from judge_eval.harness.run import main as run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="judge-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate-dataset", help="Generate the labeled faithfulness-judge evaluation dataset."
    )
    generate.add_argument("--n-samples", type=int, default=50)

    evaluate = subparsers.add_parser(
        "run-evaluation", help="Run a faithfulness judge over the generated dataset and score it."
    )
    evaluate.add_argument("--judge", choices=list(JUDGE_BACKENDS), default="custom")
    evaluate.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    evaluate.add_argument("--split", choices=["opt", "eval", "all"], default="all")

    return parser


def main() -> None:
    enable_autologging()
    args = build_parser().parse_args()

    if args.command == "generate-dataset":
        generate_dataset(args.n_samples)
    elif args.command == "run-evaluation":
        run_evaluation(args.judge, args.judge_model, args.split)


if __name__ == "__main__":
    main()
