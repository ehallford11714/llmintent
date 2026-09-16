"""Render charts from saved suite_trace report.json folders (no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Import plotter from the tracer.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from suite_trace_1000 import _save_charts, _report_md  # noqa: E402


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/suite_27b_1000")
    n = 0
    for report in sorted(root.glob("p*/report.json")):
        payload = json.loads(report.read_text(encoding="utf-8"))
        charts = _save_charts(report.parent, payload)
        payload["charts"] = [Path(p).name for p in charts]
        report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (report.parent / "report.md").write_text(_report_md(payload), encoding="utf-8")
        n += 1
    print(f"rendered charts for {n} prompt dirs in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
