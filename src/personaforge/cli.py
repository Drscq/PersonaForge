from __future__ import annotations

import argparse
import sys

from personaforge.pipeline import run_demo


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="personaforge",
        description="Population-grounded user simulation and synthetic preference-data pipeline.",
    )
    subparsers = parser.add_subparsers(required=True)

    demo = subparsers.add_parser("demo", help="Run the offline pipeline demo.")
    demo.add_argument("--source", default="sample", help="sample or path to user-log JSONL.")
    demo.add_argument("--out", default="runs/demo", help="Output directory.")
    demo.add_argument("--rounds", type=int, default=5, help="Calibration rounds.")
    demo.add_argument("--samples-per-round", type=int, default=64, help="Synthetic samples per round.")
    demo.add_argument("--seed", type=int, default=7, help="Random seed.")
    demo.set_defaults(func=cmd_demo)
    return parser


def cmd_demo(args: argparse.Namespace) -> int:
    run_demo(
        out_dir=args.out,
        source=args.source,
        rounds=args.rounds,
        samples_per_round=args.samples_per_round,
        seed=args.seed,
    )
    print(f"Wrote PersonaForge demo artifacts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

