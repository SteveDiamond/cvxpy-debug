"""Command-line interface for cvxpy-debug."""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
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
  cvxpy-debug problem.cvx                    Debug a CVX problem file
  cvxpy-debug problem.cvx --format=json      Output as JSON
  cvxpy-debug problem.cvx --format=html -o report.html
  cvxpy-debug problem.cvx --save=debugged.cvx  Save problem with debug report
  cvxpy-debug --version                      Show version

File formats:
  .cvx  CVX format (recommended, human-readable JSON)
  .pkl  Pickle format (deprecated, will be removed in future)
        """,
    )

    parser.add_argument(
        "problem_file",
        nargs="?",
        help="Path to a CVXPY Problem file (.cvx or .pkl)",
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

    parser.add_argument(
        "--save",
        metavar="FILE",
        help="Save problem and debug report to a .cvx file",
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

    import cvxpy as cp

    # Determine file type and load accordingly
    suffix = problem_path.suffix.lower()

    if suffix == ".cvx":
        # Load CVX format (recommended)
        try:
            from cvxpy_debug.format import load_cvx

            problem = load_cvx(problem_path)
        except Exception as e:
            print(f"Error loading CVX file: {e}", file=sys.stderr)
            return 1

    elif suffix == ".pkl":
        # Load pickle format (deprecated)
        warnings.warn(
            "Pickle format (.pkl) is deprecated and will be removed in a future version. "
            "Use .cvx format instead: cvxpy_debug.save_cvx(problem, 'file.cvx')",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            with open(problem_path, "rb") as f:
                problem = pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file: {e}", file=sys.stderr)
            return 1

        # Validate it's a CVXPY Problem
        if not isinstance(problem, cp.Problem):
            print("Error: File does not contain a CVXPY Problem object", file=sys.stderr)
            print(f"Got: {type(problem).__name__}", file=sys.stderr)
            return 1

    else:
        print(f"Error: Unsupported file format: {suffix}", file=sys.stderr)
        print("Supported formats: .cvx (recommended), .pkl (deprecated)", file=sys.stderr)
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

    # Save to CVX if requested
    if parsed.save:
        from cvxpy_debug.format import save_cvx

        save_path = Path(parsed.save)
        # Ensure .cvx extension
        if save_path.suffix.lower() != ".cvx":
            save_path = save_path.with_suffix(".cvx")

        save_cvx(
            problem,
            save_path,
            metadata={
                "source": str(problem_path),
                "debug_status": report.status,
            },
        )
        print(f"Problem saved to: {save_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
