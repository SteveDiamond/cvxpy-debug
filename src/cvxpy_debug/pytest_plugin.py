"""Pytest plugin for automatic CVXPY problem debugging on test failures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser
    from _pytest.nodes import Item
    from _pytest.reports import TestReport
    from _pytest.terminal import TerminalReporter

# Store problems that were solved during test execution
_solved_problems: dict[str, Any] = {}
# Store debug reports for terminal output
_debug_reports: dict[str, list] = {}


def pytest_addoption(parser: Parser) -> None:
    """Add cvxpy-debug options to pytest."""
    group = parser.getgroup("cvxpy-debug")
    group.addoption(
        "--cvxpy-debug",
        action="store_true",
        default=False,
        help="Enable automatic CVXPY problem debugging on test failures",
    )
    group.addoption(
        "--cvxpy-debug-verbose",
        action="store_true",
        default=False,
        help="Show detailed CVXPY debug output",
    )


def pytest_configure(config: Config) -> None:
    """Register the cvxpy_debug marker."""
    config.addinivalue_line(
        "markers",
        "cvxpy_debug: Mark test for automatic CVXPY debugging on failure",
    )


def pytest_runtest_makereport(item: Item, call: Any) -> TestReport | None:
    """Hook to detect CVXPY problem failures and add debug info."""
    # Only process during the call phase (not setup/teardown)
    if call.when != "call":
        return None

    # Check if test failed
    if call.excinfo is None:
        return None

    # Check if cvxpy-debug is enabled globally or via marker
    cvxpy_debug_enabled = item.config.getoption("--cvxpy-debug", default=False)
    has_marker = item.get_closest_marker("cvxpy_debug") is not None

    if not (cvxpy_debug_enabled or has_marker):
        return None

    # Try to find CVXPY problems registered via fixture
    try:
        import cvxpy as cp

        from cvxpy_debug import debug

        problems_to_debug = []

        # Get problems registered via the fixture
        if item.nodeid in _solved_problems:
            problems_to_debug.extend(_solved_problems[item.nodeid])

        # Debug any non-optimal problems found
        for prob in problems_to_debug:
            if isinstance(prob, cp.Problem) and prob.status is not None:
                if prob.status != cp.OPTIMAL:
                    report = debug(prob, verbose=False)
                    # Store the report globally for terminal output
                    if item.nodeid not in _debug_reports:
                        _debug_reports[item.nodeid] = []
                    _debug_reports[item.nodeid].append(report)

    except Exception:
        # Don't let debugging errors affect test execution
        pass

    return None


@pytest.hookimpl(tryfirst=True)
def pytest_report_teststatus(report: TestReport, config: Config) -> tuple[str, str, str] | None:
    """Add CVXPY debug info to test report."""
    return None


def pytest_terminal_summary(
    terminalreporter: TerminalReporter,
    exitstatus: int,
    config: Config,
) -> None:
    """Print CVXPY debug reports at the end of the test session."""
    if not _debug_reports:
        return

    terminalreporter.write_sep("=", "CVXPY Debug Reports")
    for nodeid, reports in _debug_reports.items():
        terminalreporter.write_line(f"\n{nodeid}:")
        for report in reports:
            terminalreporter.write_line(str(report))

    # Clear for next run
    _debug_reports.clear()


@pytest.fixture
def cvxpy_debug_tracker(request: pytest.FixtureRequest):
    """
    Fixture to track CVXPY problems for debugging.

    Usage::

        def test_my_optimization(cvxpy_debug_tracker):
            x = cp.Variable()
            prob = cp.Problem(cp.Minimize(x), [x >= 10, x <= 5])
            cvxpy_debug_tracker(prob)  # Register problem for debugging
            prob.solve()
            assert prob.status == cp.OPTIMAL  # Will trigger debug on failure
    """

    def tracker(problem: Any) -> Any:
        """Register a problem for debugging on test failure."""
        if request.node.nodeid not in _solved_problems:
            _solved_problems[request.node.nodeid] = []
        _solved_problems[request.node.nodeid].append(problem)
        return problem

    yield tracker

    # Cleanup
    if request.node.nodeid in _solved_problems:
        del _solved_problems[request.node.nodeid]
