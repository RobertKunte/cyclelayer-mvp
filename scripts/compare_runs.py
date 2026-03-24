"""Compare metrics from two run directories (e.g. CNN baseline vs CycleLayer).

Loads metrics_dev.json and metrics_test.json from each run dir, prints a
structured comparison table to stdout, and saves comparison.md.

Usage
-----
    python scripts/compare_runs.py \\
        --run_a  runs/20240101_120000_cnn \\
        --run_b  runs/20240101_140000_cyclelayer_v1 \\
        --label_a  "CNN" \\
        --label_b  "CycleLayer" \\
        --out_dir  runs/comparison_20240101

Output
------
    {out_dir}/comparison.md   -- Markdown pipe-table document
    stdout                    -- same table in plain text

Metric conventions
------------------
    Δ  = label_b − label_a   (positive Δ = B worse for lower-is-better metrics)
    ✓  = label_b (second model) is better
    ✗  = label_a (first model) is better or equal
    Missing keys show as "n/a" and are excluded from delta/flag columns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Metric registry — (json_key, display_label, lower_is_better)
# ---------------------------------------------------------------------------
METRICS: list[tuple[str, str, bool]] = [
    ("rmse",                          "RMSE",               True),
    ("rmse_cycle",                    "RMSE (cycle-avg)",   True),
    ("s_score_mean",                  "S-score/win",        True),
    ("max_abs_error_unit_median",     "MaxAbsErr median",   True),
    ("ph_median",                     "PH median",          False),
    ("ph_none_count",                 "PH none",            True),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_metrics(run_dir: Path, split: str) -> dict[str, Any]:
    """Try metrics_{split}.json, fallback metrics.json.  Returns {} on failure."""
    for name in (f"metrics_{split}.json", "metrics.json"):
        p = run_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[WARN] Could not parse {p}: {exc}", file=sys.stderr)
    return {}


def _fmt(v: Any) -> str:
    """Format a single metric value for display."""
    if v is None:
        return "n/a"
    try:
        f = float(v)
        if f != f:  # NaN
            return "n/a"
        # Integer-valued floats: show without decimals
        if abs(f) >= 1 and f == int(f) and abs(f) < 1e6:
            return str(int(f))
        return f"{f:.4f}"
    except (TypeError, ValueError):
        return str(v)


def _delta_row(v_a: Any, v_b: Any) -> tuple[str, str, float | None]:
    """Return (delta_str, delta_pct_str, raw_delta) or ("—", "—", None)."""
    try:
        a = float(v_a)
        b = float(v_b)
        if a != a or b != b:  # NaN guard
            return "—", "—", None
        delta = b - a
        denom = abs(a) if a != 0 else abs(b)
        pct = 100.0 * delta / denom if denom != 0 else float("nan")
        pct_str = f"{pct:+.1f}%" if pct == pct else "—"
        return f"{delta:+.4f}", pct_str, delta
    except (TypeError, ValueError):
        return "—", "—", None


def _flag(raw_delta: float | None, lower_is_better: bool) -> str:
    if raw_delta is None:
        return "—"
    return "✓" if (lower_is_better and raw_delta < 0) or (not lower_is_better and raw_delta > 0) else "✗"


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def _build_rows(
    m_a: dict[str, Any],
    m_b: dict[str, Any],
    split: str,
) -> list[tuple[str, str, str, str, str, str]]:
    rows = []
    for key, label, lower in METRICS:
        v_a = m_a.get(key)
        v_b = m_b.get(key)
        if v_a is None and v_b is None:
            continue
        delta_s, pct_s, raw = _delta_row(v_a, v_b)
        flag = _flag(raw, lower)
        rows.append((f"{split}/{label}", _fmt(v_a), _fmt(v_b), delta_s, pct_s, flag))
    return rows


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _render_plain(rows: list[tuple], label_a: str, label_b: str) -> str:
    col_w = [26, 11, 11, 12, 8, 4]
    hdrs = ["Split / Metric", label_a[:11], label_b[:11], "Δ", "Δ%", "✓?"]

    def _row(cells: list[str]) -> str:
        return "  ".join(c.ljust(col_w[i]) for i, c in enumerate(cells))

    sep = "  ".join("-" * w for w in col_w)
    lines = [_row(hdrs), sep]
    prev_split = None
    for r in rows:
        s = r[0].split("/")[0]
        if prev_split is not None and s != prev_split:
            lines.append(sep)
        lines.append(_row(list(r)))
        prev_split = s

    footer = (
        f"\nΔ = {label_b} − {label_a}  |  ✓ = {label_b} better  |  ✗ = {label_a} better or equal\n"
        "lower is better: RMSE, S-score/win, MaxAbsErr, PH none\n"
        "higher is better: PH median"
    )
    return "\n".join(lines) + "\n" + footer


def _render_markdown(
    rows: list[tuple],
    label_a: str,
    label_b: str,
    run_a: Path,
    run_b: Path,
) -> str:
    def _pipe(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    sep_row = "| " + " | ".join(["---"] * 6) + " |"
    hdr_row = _pipe(["Split / Metric", label_a, label_b, "Δ", "Δ%", "✓?"])

    table_lines = [hdr_row, sep_row]
    prev_split = None
    for r in rows:
        s = r[0].split("/")[0]
        if prev_split is not None and s != prev_split:
            table_lines.append(_pipe(["", "", "", "", "", ""]))  # blank separator row
        table_lines.append(_pipe(list(r)))
        prev_split = s

    return f"""\
# Model Comparison

| | |
|---|---|
| **Model A ({label_a})** | `{run_a}` |
| **Model B ({label_b})** | `{run_b}` |

## Metrics

{chr(10).join(table_lines)}

**Notes**

- Δ = {label_b} − {label_a}
- ✓ = {label_b} (B) is better; ✗ = {label_a} (A) is better or equal
- Lower is better: RMSE, RMSE (cycle-avg), S-score/win, MaxAbsErr median, PH none
- Higher is better: PH median
- `n/a` = metric not present in that run's metrics JSON
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare metrics from two run directories. Prints table and saves comparison.md."
    )
    parser.add_argument("--run_a",   required=True,  help="First run directory (e.g. CNN baseline).")
    parser.add_argument("--run_b",   required=True,  help="Second run directory (e.g. CycleLayer).")
    parser.add_argument("--label_a", default="Model A", help="Display label for run_a.")
    parser.add_argument("--label_b", default="Model B", help="Display label for run_b.")
    parser.add_argument(
        "--out_dir", default=None,
        help="Directory for comparison.md. Defaults to <run_a>/../comparison/.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["dev", "test"],
        choices=["dev", "test"],
        help="Splits to include (default: dev test).",
    )
    args = parser.parse_args()

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)

    for p, flag in [(run_a, "--run_a"), (run_b, "--run_b")]:
        if not p.exists():
            print(f"[ERROR] {flag} directory does not exist: {p}", file=sys.stderr)
            sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else run_a.parent / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[tuple] = []
    for split in args.splits:
        m_a = _load_metrics(run_a, split)
        m_b = _load_metrics(run_b, split)
        if not m_a and not m_b:
            print(f"[WARN] No metrics for split '{split}' in either run dir.", file=sys.stderr)
            continue
        all_rows.extend(_build_rows(m_a, m_b, split))

    if not all_rows:
        print(
            "[ERROR] No comparable metrics found. "
            "Ensure metrics_dev.json or metrics_test.json exist in both run dirs.",
            file=sys.stderr,
        )
        sys.exit(1)

    plain = _render_plain(all_rows, args.label_a, args.label_b)
    width = 76
    print()
    print("=" * width)
    print(f"COMPARISON  ·  {args.label_a}  vs  {args.label_b}")
    print("=" * width)
    print(plain)
    print("=" * width)
    print(f"\nRun A ({args.label_a}): {run_a}")
    print(f"Run B ({args.label_b}): {run_b}")

    md = _render_markdown(all_rows, args.label_a, args.label_b, run_a, run_b)
    md_path = out_dir / "comparison.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"\nSaved: {md_path}")


if __name__ == "__main__":
    main()
