"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from causal_workspace_jepa.common.resources import inspect_resources


def _doctor(args: argparse.Namespace) -> int:
    report = inspect_resources(args.resource_profile, args.root)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.ok else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="causal_workspace_jepa")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="Check local resources against a profile")
    doctor.add_argument("--resource-profile", required=True)
    doctor.add_argument("--root", default=".")
    doctor.set_defaults(func=_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
