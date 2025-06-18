"""Simple reporting utilities for manipulation analysis results."""

import csv
import html
import json
from typing import Any, Dict


def generate_summary_report(scoring_data: Dict[str, Any], reliability_data: Dict[str, Any]) -> str:
    """Generate a readable text summary for provided data."""
    summary_lines = ["Manipulation Detection Report"]
    summary_lines.append("== Scoring ==")
    for k, v in scoring_data.items():
        summary_lines.append(f"{k}: {v}")
    summary_lines.append("\n== Reliability ==")
    for k, v in reliability_data.items():
        summary_lines.append(f"{k}: {v}")
    return "\n".join(summary_lines)


def export_report(data: Dict[str, Any], format: str = "json", path: str = "report") -> None:
    """Export data to disk in the given format."""
    if format == "json":
        with open(f"{path}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif format == "csv":
        with open(f"{path}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for key, value in data.items():
                writer.writerow([key, value])
    elif format == "html":
        with open(f"{path}.html", "w", encoding="utf-8") as f:
            f.write("<html><body><pre>")
            f.write(html.escape(json.dumps(data, indent=2)))
            f.write("</pre></body></html>")
    else:
        raise ValueError(f"Unsupported format: {format}")
