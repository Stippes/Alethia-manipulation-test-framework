"""Generate a simple HTML dashboard summarizing pytest results."""

import os
import sys
import json
import subprocess
from html import escape
from pathlib import Path
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logging_utils import setup_logging

logger = logging.getLogger(__name__)


def run_tests(json_path: str = "test_results.json") -> None:
    """Run pytest and write a JSON report to ``json_path``."""
    subprocess.run(
        ["pytest", "-q", "--json-report", f"--json-report-file={json_path}"],
        check=False,
    )


def build_dashboard(json_path: str = "test_results.json", html_path: str = "dashboard.html") -> None:
    """Create an HTML dashboard from a pytest JSON report."""
    data = json.loads(Path(json_path).read_text())
    lines = ["<html><head><title>Test Results</title></head><body>"]
    lines.append("<h1>Test Results</h1>")
    summary = data.get("summary", {})
    lines.append(
        f"<p>Passed: {summary.get('passed', 0)} / {summary.get('total', 0)}<br>"
        f"Collected: {summary.get('collected', 0)}</p>"
    )
    lines.append("<table border='1' cellpadding='5' cellspacing='0'>")
    lines.append("<tr><th>Test</th><th>Outcome</th></tr>")
    for test in data.get("tests", []):
        name = escape(test.get("nodeid", ""))
        outcome = escape(test.get("outcome", ""))
        lines.append(f"<tr><td>{name}</td><td>{outcome}</td></tr>")
    lines.append("</table>")
    lines.append("</body></html>")
    Path(html_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_logging()
    run_tests()
    build_dashboard()
    logger.info("Dashboard written to dashboard.html")


if __name__ == "__main__":
    main()

