#!/usr/bin/env python3
"""Validate, inspect, and render the unified research project state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.project_state import (  # noqa: E402
    DEFAULT_README,
    DEFAULT_REGISTRY,
    DEFAULT_STATUS_DOCUMENT,
    ProjectStateError,
    check_outputs,
    current_snapshot,
    load_registry,
    render_agent_summary,
    render_status_markdown,
    validate_registry,
    write_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="canonical registry JSON (default: research_state/registry.json)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "validate", help="lightweight schema, links, and state-invariant check"
    )

    show = subparsers.add_parser("show", help="print the current projection without writing")
    show.add_argument(
        "--format",
        choices=("agent", "json", "markdown"),
        default="agent",
    )

    render = subparsers.add_parser("render", help="write or check generated status views")
    render.add_argument(
        "--check",
        action="store_true",
        help="fail when generated views are stale; do not write",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry_path = args.registry.resolve()
        registry = load_registry(registry_path)
        validate_registry(registry, repo_root=REPO_ROOT)

        if args.command == "validate":
            print(
                f"valid {registry['schema']}: "
                f"{len(registry['records'])} records, "
                f"{len(current_snapshot(registry)['records'])} current entities"
            )
            return 0

        if args.command == "show":
            if args.format == "agent":
                sys.stdout.write(render_agent_summary(registry))
            elif args.format == "json":
                print(
                    json.dumps(
                        current_snapshot(registry),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                sys.stdout.write(render_status_markdown(registry, repo_root=REPO_ROOT))
            return 0

        status_path = DEFAULT_STATUS_DOCUMENT
        readme_path = DEFAULT_README
        if args.check:
            stale = check_outputs(
                registry,
                repo_root=REPO_ROOT,
                status_path=status_path,
                readme_path=readme_path,
            )
            if stale:
                for path in stale:
                    print(f"stale generated project-state view: {path}", file=sys.stderr)
                print(
                    "run `.venv/bin/python experiments/scripts/project_state.py render`",
                    file=sys.stderr,
                )
                return 1
            print("generated project-state views are current")
            return 0

        for path in write_outputs(
            registry,
            repo_root=REPO_ROOT,
            status_path=status_path,
            readme_path=readme_path,
        ):
            print(f"updated {path.relative_to(REPO_ROOT)}")
        return 0
    except (OSError, ProjectStateError) as exc:
        print(f"project-state error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
