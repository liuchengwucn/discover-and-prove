"""Command-line entry point for the Discover-and-Prove (DAP) framework.

Subcommands
-----------
main               Run the full DAP pipeline (Table 1). Use --no-agent for the
                   ablation without self-verification/self-correction.
answer-acc         Judge Discovery-Module answer accuracy against ground truth
                   on an existing results file (Table 3).
rewrite-ablation   Run the rewriting-strategy ablation (Table 5):
                   --mode {ours,straight,none}.

Examples
--------
    python run.py main --dataset putnam                 # DAP w/ Agent
    python run.py main --dataset putnam --no-agent      # DAP w/o Agent
    python run.py answer-acc --dataset putnam \
        --results eval_dap_on_putnam_intermediate_1.0.json
    python run.py rewrite-ablation --dataset putnam --mode none
    python run.py rewrite-ablation --dataset putnam --mode straight
"""

import argparse

from discover_and_prove import main as run_pipeline
from judge import evaluate_answer_accuracy

# CLI dataset names -> the dataset identifiers used by the pipeline loader.
PIPELINE_DATASET = {
    "putnam": "putnam",
    "combi": "combi",
    "minif2f": "minif2f_annotated",
    "fimo": "fimo_annotated",
}
DATASETS = list(PIPELINE_DATASET.keys())


def _add_dataset_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Fraction of the dataset to run (for quick smoke tests).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover and Prove (DAP) runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_main = sub.add_parser("main", help="Run the full DAP pipeline (Table 1).")
    _add_dataset_arg(p_main)
    p_main.add_argument(
        "--no-agent",
        action="store_true",
        help="Disable self-verification/self-correction (DAP w/o Agent).",
    )

    p_acc = sub.add_parser(
        "answer-acc", help="Judge Discovery answer accuracy (Table 3)."
    )
    p_acc.add_argument("--dataset", required=True, choices=DATASETS)
    p_acc.add_argument(
        "--results", required=True, help="Results JSON produced by `run.py main`."
    )
    p_acc.add_argument(
        "--judge",
        default="gemini",
        choices=["gemini", "gpt-oss"],
        help="Judge model. The paper uses gemini-2.5-flash.",
    )

    p_abl = sub.add_parser(
        "rewrite-ablation", help="Rewriting-strategy ablation (Table 5)."
    )
    _add_dataset_arg(p_abl)
    p_abl.add_argument(
        "--mode",
        required=True,
        choices=["ours", "straight", "none"],
        help="ours = discover then rewrite; straight = single-step rewrite; "
        "none = feed the Hard Mode statement directly.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "main":
        run_pipeline(
            dataset=PIPELINE_DATASET[args.dataset],
            ratio=args.ratio,
            no_agent=args.no_agent,
            rewrite_mode="ours",
        )
    elif args.command == "rewrite-ablation":
        run_pipeline(
            dataset=PIPELINE_DATASET[args.dataset],
            ratio=args.ratio,
            no_agent=False,
            rewrite_mode=args.mode,
        )
    elif args.command == "answer-acc":
        evaluate_answer_accuracy(args.results, args.dataset, judge=args.judge)


if __name__ == "__main__":
    main()
