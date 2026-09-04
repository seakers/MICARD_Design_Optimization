"""CLI for a MICARD design-exploration run -- both the human-facing default
and the machine-readable `--json` mode an embedding caller (e.g. design_team's
design_optimization Tool) invokes via subprocess against this package's own
venv. A subprocess, not a direct import, because that venv is deliberately
separate from the caller's own (see README.md's "Install" section) -- this
CLI is the only contract between them.

`--json` prints exactly one line on stdout: run()'s result as JSON. Anything
else this pipeline writes to stdout (this module's own logging, and a few
progress print()s down in the optimizers/catalog loader) is redirected to
stderr for the duration of the run so it can never end up mixed into that
line -- a caller in --json mode should read stdout as pure JSON and stderr
as diagnostics.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from micard_design_optimization.run import run

_METHOD_CHOICES = ("random_search", "genetic_algorithm", "ppo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-path", default=None,
                        help="JSON parts catalog (default: the bundled Dynamixel catalog)")
    parser.add_argument("--max-joints", type=int, default=6)
    parser.add_argument("--n-ports", type=int, default=7)
    parser.add_argument("--random-evals", type=int, default=None,
                        help="default: ga_pop * ga_gen")
    parser.add_argument("--ga-pop", type=int, default=100)
    parser.add_argument("--ga-gen", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None, choices=_METHOD_CHOICES,
                        help="default: all three")
    parser.add_argument("--output-root", default="designs",
                        help="where the session folder is created (default: ./designs)")
    parser.add_argument("--json", action="store_true",
                        help="print run()'s result as one JSON line on stdout instead of "
                             "a human-readable summary; see this module's own docstring")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # logging.basicConfig's default stream is already stderr, but the odd
    # stray print() further down (catalog loading, per-generation optimizer
    # progress) is not -- see the redirect below for --json.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    kwargs = dict(
        catalog_path=args.catalog_path,
        max_joints=args.max_joints,
        n_ports=args.n_ports,
        random_evals=args.random_evals,
        ga_pop=args.ga_pop,
        ga_gen=args.ga_gen,
        seed=args.seed,
        methods=args.methods,
        output_root=args.output_root,
    )

    if args.json:
        real_stdout = sys.stdout
        sys.stdout = sys.stderr  # catch any stray print(), not just logging
        try:
            result = run(**kwargs)
        finally:
            sys.stdout = real_stdout
        print(json.dumps(result))
        return

    result = run(**kwargs)
    print(f"\nSession {result['session_id']}: {result['session_root']}")


if __name__ == "__main__":
    main()
