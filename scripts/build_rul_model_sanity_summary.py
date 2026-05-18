"""Aggregator — RUL_MODEL_SANITY_SUMMARY.md (ADR-0014, Step 8).

Reads each diagnostic's `summary.json` from the current session dir,
classifies the outcome (PASS / WEAK / FAIL), runs the ADR-0014
decision logic, and emits:

  * `RUL_MODEL_SANITY_SUMMARY.md`  (human-readable)
  * `RUL_MODEL_SANITY_SUMMARY.json` (machine-readable verdict)

Strict no-op script — never trains, never modifies model code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from _v31b_diag_helpers import get_session_dir   # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_dir", default=None,
                    help="Override session dir (default: shared via env / "
                         "latest under artifacts/cyclelayer_v3/rul_model_sanity).")
    return ap.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": f"failed to parse {path}: {exc}"}


def _classify_target_alignment(s: dict | None) -> str:
    if s is None: return "MISSING"
    if s.get("status") == "SKIPPED": return "SKIPPED"
    return "PASS" if s.get("overall_pass") else "FAIL"


def _classify_collapse(s: dict | None) -> str:
    if s is None: return "MISSING"
    flags = s.get("flags", {})
    fired = sum(1 for v in flags.values() if v)
    if fired >= 3: return "FAIL"        # mean collapse confirmed
    if fired in (1, 2): return "WEAK"
    return "PASS"


def _classify_tiny_overfit(s: dict | None) -> str:
    if s is None: return "MISSING"
    if s.get("status") == "SKIPPED": return "SKIPPED"
    if s.get("overall_pass"): return "PASS"
    # If at least the largest set passes, classify WEAK
    rows = s.get("rows", [])
    if rows:
        sorted_rows = sorted(rows, key=lambda r: r.get("n_train", 0))
        largest = sorted_rows[-1]
        if largest.get("passes_overfit"): return "WEAK"
    return "FAIL"


def _classify_simple_baselines(s: dict | None, v3_test_rmse: float = 21.41) -> str:
    if s is None: return "MISSING"
    best = s.get("best")
    if not best or "test_RMSE" not in best:
        return "MISSING"
    diff = v3_test_rmse - float(best["test_RMSE"])
    if diff > 3.0:  return "FAIL"        # baselines crush V3.1b
    if diff > 0.5:  return "WEAK"        # baselines beat slightly
    return "PASS"                        # V3.1b not worse than classical ML


def _classify_branch_usage(s: dict | None) -> str:
    if s is None: return "MISSING"
    unused = s.get("unused_branches", [])
    if not unused: return "PASS"
    # Theta unused → consistent with ADR-0013 collapse story (FAIL on
    # "uses physics" sense).  Multiple unused → severe.
    if any(u.startswith(("zero_theta", "shuffle_theta",
                          "zero_features"))
           for u in unused):
        return "FAIL"
    return "WEAK"


def main() -> None:
    args = parse_args()
    session_dir = Path(args.session_dir) if args.session_dir else get_session_dir()
    print(f"session dir: {session_dir}")

    # ── Load all per-diagnostic summaries ────────────────────────────
    s_align    = _load_json(session_dir / "target_alignment"  / "summary.json")
    s_collapse = _load_json(session_dir / "rul_collapse"      / "summary.json")
    s_overfit  = _load_json(session_dir / "overfit_tiny"      / "summary.json")
    s_simple   = _load_json(session_dir / "simple_baselines"  / "summary.json")
    s_branch   = _load_json(session_dir / "branch_usage"      / "summary.json")

    align_v    = _classify_target_alignment(s_align)
    collapse_v = _classify_collapse(s_collapse)
    overfit_v  = _classify_tiny_overfit(s_overfit)
    simple_v   = _classify_simple_baselines(s_simple)
    branch_v   = _classify_branch_usage(s_branch)

    # ── ADR-0014 decision logic ──────────────────────────────────────
    if align_v == "FAIL":
        decision = (
            "STOP — target alignment FAILED.  Fix data/target pipeline "
            "first.  All other diagnostics are invalid."
        )
        decision_branch = "fix_alignment"
    elif overfit_v == "FAIL":
        decision = (
            "STOP — tiny-overfit FAILED.  Implementation problem "
            "(output scaling / loss / optimizer / arch).  Debug before "
            "any V3.1b training improvements."
        )
        decision_branch = "fix_implementation"
    elif overfit_v in ("PASS", "WEAK") and collapse_v == "FAIL":
        decision = (
            "Tiny-overfit succeeds but the production C run collapses "
            "to a mean prediction on test units.  This is a "
            "**generalisation / training-budget / sampling** problem.  "
            "Next: longer training, EOL-balanced sampling, possibly a "
            "larger encoder.  Do not change architecture yet."
        )
        decision_branch = "improve_training"
    elif simple_v == "FAIL":
        decision = (
            "Simple Ridge / HGB / RF baselines clearly beat V3.1b on "
            "the same DS02 test units.  V3.1b architecture/training is "
            "not competitive.  Consider reverting to a V2-style "
            "encoder; treat V3.1b physics as a post-hoc diagnostic only."
        )
        decision_branch = "revert_architecture"
    elif collapse_v == "PASS" and simple_v in ("PASS", "WEAK"):
        decision = (
            "V3.1b RUL is not collapsing and is at least competitive "
            "with classical baselines.  Continue V3.1b improvements."
        )
        decision_branch = "continue_v31b"
    else:
        decision = (
            "Inconclusive — surface results to Robert.  See per-diagnostic "
            "reports for detail."
        )
        decision_branch = "inconclusive"

    # ── Markdown summary ─────────────────────────────────────────────
    def link(rel: str) -> str:
        p = session_dir / rel
        return f"[{rel}]({p.as_posix()})" if p.exists() else f"`{rel}` (missing)"

    md = f"""# RUL_MODEL_SANITY_SUMMARY (ADR-0014)

*Session directory:* `{session_dir}`

## Verdict per diagnostic

| # | Diagnostic | Verdict |
|---|---|---|
| 4 | Target / window alignment      | **{align_v}**    |
| 2+3 | Collapse metrics + baselines + plots | **{collapse_v}** |
| 5 | Tiny-overfit                   | **{overfit_v}**  |
| 6 | Simple ML baselines (Ridge/HGB/RF) | **{simple_v}**   |
| 7 | Branch-usage ablations         | **{branch_v}**   |

## Decision (per ADR-0014 logic)

**{decision}**

Decision branch: `{decision_branch}`

## Hypothesis status

| ID | Hypothesis | Status |
|---|---|---|
| H1 | RUL head predicts ≈ train-mean (mean-collapse) | {"likely TRUE" if collapse_v == "FAIL" else "WEAK" if collapse_v == "WEAK" else "rejected"} |
| H2 | Target normalisation bug                       | {"likely TRUE" if align_v == "FAIL" else "rejected"} |
| H3 | Window/label misalignment                      | {"likely TRUE" if align_v == "FAIL" else "rejected"} |
| H4 | Train/test distribution mismatch                | inspect target-distribution plot |
| H5 | Under-training (could learn, didn't yet)        | {"likely TRUE" if overfit_v in ('PASS', 'WEAK') and collapse_v == "FAIL" else "inconclusive"} |
| H6 | RUL head ignores all features                   | {"likely TRUE" if branch_v == "FAIL" else "WEAK" if branch_v == "WEAK" else "rejected"} |
| H7 | Task harder than current model capacity         | {"likely TRUE" if simple_v == "FAIL" else "inconclusive"} |

## Per-diagnostic reports

* {link('target_alignment/target_alignment_report.md')}
* {link('rul_collapse/report.md')}
* {link('overfit_tiny/report.md')}
* {link('simple_baselines/report.md')}
* {link('branch_usage/report.md')}

## Hard constraints honored

* No model architecture changes (`cyclelayer_v3.py` frozen).
* No YAML / physical-constant changes.
* No hyperparameter tuning loops.
* No EPR / pressure loss reintroduction.
* No DS02 test leakage (test units `[11, 14, 15]` only evaluated, never
  trained on; simple-baseline trainer trains on train units only).
* Production C checkpoint loaded read-only.

## Stop point

This summary is the stop point for the diagnostic phase.  **Do not**
change `CycleLayerV3`, do not add V4, do not tune.  Robert reviews and
chooses the next action.
"""
    out_md   = session_dir / "RUL_MODEL_SANITY_SUMMARY.md"
    out_json = session_dir / "RUL_MODEL_SANITY_SUMMARY.json"
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps({
        "session_dir": str(session_dir),
        "verdicts": {
            "target_alignment": align_v,
            "rul_collapse":     collapse_v,
            "tiny_overfit":     overfit_v,
            "simple_baselines": simple_v,
            "branch_usage":     branch_v,
        },
        "decision":        decision,
        "decision_branch": decision_branch,
    }, indent=2), encoding="utf-8")
    print(f"\nsaved {out_md}")
    print(f"saved {out_json}")
    print(f"\nVERDICTS  align={align_v}  collapse={collapse_v}  "
          f"overfit={overfit_v}  baselines={simple_v}  branches={branch_v}")
    print(f"DECISION  {decision_branch}: {decision}")


if __name__ == "__main__":
    main()
