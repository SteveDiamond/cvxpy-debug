"""Command-line interface for cvxpy-debug."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def main(args: list[str] | None = None) -> int:
    """
    Main entry point for the cvxpy-debug CLI.

    Parameters
    ----------
    args : list[str] | None, optional
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        prog="cvxpy-debug",
        description="Diagnostic tool for CVXPY optimization problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cvxpy-debug problem.pkl                    Debug a pickled problem (text output)
  cvxpy-debug problem.pkl --format=json      Output as JSON
  cvxpy-debug problem.pkl --format=html -o report.html
  cvxpy-debug --version                      Show version
        """,
    )

    parser.add_argument(
        "problem_file",
        nargs="?",
        help="Path to a pickled CVXPY Problem (.pkl file)",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["text", "json", "html"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: stdout for text/json, required for html)",
    )

    parser.add_argument(
        "--no-iis",
        action="store_true",
        help="Skip minimal IIS refinement (faster)",
    )

    parser.add_argument(
        "--no-conditioning",
        action="store_true",
        help="Skip condition number analysis (faster for large problems)",
    )

    parser.add_argument(
        "--no-performance",
        action="store_true",
        help="Skip performance analysis",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    parsed = parser.parse_args(args)

    # Handle version flag
    if parsed.version:
        from cvxpy_debug import __version__

        print(f"cvxpy-debug {__version__}")
        return 0

    # Check for required problem file
    if not parsed.problem_file:
        parser.print_help()
        return 1

    # Load the problem file
    problem_path = Path(parsed.problem_file)
    if not problem_path.exists():
        print(f"Error: File not found: {problem_path}", file=sys.stderr)
        return 1

    try:
        with open(problem_path, "rb") as f:
            problem = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}", file=sys.stderr)
        return 1

    # Validate it's a CVXPY Problem
    import cvxpy as cp

    if not isinstance(problem, cp.Problem):
        print("Error: File does not contain a CVXPY Problem object", file=sys.stderr)
        print(f"Got: {type(problem).__name__}", file=sys.stderr)
        return 1

    # Run the debug
    from cvxpy_debug import debug

    report = debug(
        problem,
        verbose=False,
        find_minimal_iis=not parsed.no_iis,
        include_conditioning=not parsed.no_conditioning,
        include_performance=not parsed.no_performance,
    )

    # Output the report
    if parsed.format == "text":
        output = str(report)
        if parsed.output:
            Path(parsed.output).write_text(output, encoding="utf-8")
        else:
            print(output)

    elif parsed.format == "json":
        output = report.to_json()
        if parsed.output:
            Path(parsed.output).write_text(output, encoding="utf-8")
        else:
            print(output)

    elif parsed.format == "html":
        if not parsed.output:
            # For HTML, default to stdout but warn
            print(report.to_html())
        else:
            report.to_html(file=parsed.output)
            print(f"Report saved to: {parsed.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
