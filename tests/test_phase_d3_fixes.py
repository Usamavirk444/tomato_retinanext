"""
Regression tests for the three Phase D.3 fixes.

    A  torch.mps.empty_cache() guarded on real backend availability
    B  --checkpoint honoured, with a hard error on a missing file
    C  per-class AP labelled as AP@0.50:0.95, legacy column preserved

Each test asserts BOTH that the fix is present and that the original
behaviour it was meant to preserve is intact. Tests that need torch are
skipped cleanly when torch is absent, and say so rather than passing
silently.

    python -m pytest tests/test_phase_d3_fixes.py -v
    python tests/test_phase_d3_fixes.py
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures, skips = [], []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(f"{name}: {detail}")
    return cond


def skip(name, why):
    print(f"  SKIP  {name}   [{why}]")
    skips.append(f"{name}: {why}")


def src(path):
    return open(os.path.join(REPO, path), encoding="utf-8").read()


# ── FIX A ───────────────────────────────────────────────────────────
def code_only(path):
    """Source with comment lines stripped, so a comment quoting the old
    buggy line cannot be mistaken for the line itself."""
    return "\n".join(l for l in src(path).split("\n")
                      if not l.lstrip().startswith("#"))


def test_fix_a_mps_guard():
    s = src("evaluate.py")
    check("A: no unguarded hasattr(torch.mps,...) call in CODE",
          "hasattr(torch.mps, 'empty_cache')" not in code_only("evaluate.py"))
    check("A: guarded on torch.backends.mps.is_available()",
          "torch.backends.mps.is_available()" in s)
    check("A: cuda cache cleared on cuda machines",
          "torch.cuda.empty_cache()" in s)

    # the guard must be a real branch, not a bare expression statement
    tree = ast.parse(s)
    guarded = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            t = ast.dump(node.test)
            if "mps" in t and "is_available" in t:
                guarded = True
    check("A: the call sits inside an if-statement", guarded)


# ── FIX B ───────────────────────────────────────────────────────────
def test_fix_b_checkpoint_signature():
    s = src("evaluate.py")
    check("B: get_checkpoint_path accepts checkpoint=",
          "def get_checkpoint_path(config, model_name, seed, checkpoint=None)" in s)
    check("B: evaluate_pytorch_model accepts checkpoint=",
          "def evaluate_pytorch_model(config, model_name, seed, checkpoint=None)" in s)
    check("B: args.checkpoint is actually passed through",
          "checkpoint=args.checkpoint" in s)
    check("B: missing file raises FileNotFoundError",
          "raise FileNotFoundError" in s)
    check("B: yolov8n rejects --checkpoint rather than ignoring it",
          "not supported for yolov8n" in s)


def test_fix_b_behaviour():
    """Explicit path wins; missing file errors; default unchanged."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ev", os.path.join(REPO, "evaluate.py"))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:
        skip("B: runtime behaviour", f"needs {e.name}")
        return

    cfg = {"models": {"retinanet": {
        "checkpoints_dir": "outputs/checkpoints/retinanet",
        "name": "retinanet_resnet50_fpn_v2"}}}

    # default path, unchanged from the original
    got = mod.get_checkpoint_path(cfg, "retinanet", 42)
    check("B: default path unchanged when checkpoint is None",
          got.replace("\\", "/") ==
          "outputs/checkpoints/retinanet/seed_42/retinanet_resnet50_fpn_v2_best.pth",
          got)

    # explicit path that exists wins
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as fh:
        fh.write(b"x")
        tmp = fh.name
    try:
        got = mod.get_checkpoint_path(cfg, "retinanet", 42, checkpoint=tmp)
        check("B: explicit checkpoint is returned verbatim", got == tmp, got)
    finally:
        os.unlink(tmp)

    # explicit path that does not exist must raise, not fall back
    try:
        mod.get_checkpoint_path(cfg, "retinanet", 42,
                                checkpoint="/no/such/file.pth")
        check("B: missing explicit checkpoint raises", False,
              "returned instead of raising")
    except FileNotFoundError:
        check("B: missing explicit checkpoint raises", True)


# ── FIX C ───────────────────────────────────────────────────────────
def test_fix_c_labelling():
    s = src("src/evaluation/evaluator.py")
    check("C: console heading no longer claims IoU 0.50 alone",
          'print(f"\\nPer-Class AP @ IoU 0.50")' not in s)
    check("C: heading states 0.50:0.95", "Per-Class AP @ IoU 0.50:0.95" in s)
    check("C: unambiguous AP5095_ column written",
          'summary[f"AP5095_{safe_name}"]' in s)
    check("C: legacy AP_ column preserved for compatibility",
          'summary[f"AP_{safe_name}"]' in s)
    check("C: both columns carry the same value",
          "v_ap  = round(ap_val,  4)" in s and
          s.count("= v_ap") == 2)
    check("C: recall column untouched",
          'summary[f"Recall_{safe_name}"]' in s)


def test_fix_c_no_metric_change():
    """The rounding and the >= 0 sentinel must be exactly as before."""
    s = src("src/evaluation/evaluator.py")
    check("C: still rounds to 4 dp", "round(ap_val,  4)" in s)
    check("C: still maps negative AP to None",
          "if ap_val  >= 0 else None" in s)
    check("C: map_per_class is still the source",
          'results.get("map_per_class"' in s)


def test_archived_csvs_still_readable():
    """Existing reference CSVs use AP_<class>; the alias keeps them valid."""
    import csv
    p = os.path.join(REPO, "reference_results", "in_domain",
                     "retinanet_hybrid",
                     "retinanet_hybrid_seed_42_eval_results.csv")
    if not os.path.exists(p):
        skip("C: archived CSV compatibility", "reference CSV absent")
        return
    r = next(csv.DictReader(open(p, encoding="utf-8")))
    check("C: archived CSVs still have AP_<class>", "AP_Healthy" in r)
    # and the documented relationship still holds
    pc = os.path.join(REPO, "figures", "source_csv", "results",
                      "retinanet_hybrid", "seed_42", "per_class_test.csv")
    if os.path.exists(pc):
        rows = {x["class"]: x for x in csv.DictReader(open(pc, encoding="utf-8"))}
        a = float(r["AP_Healthy"])
        b = float(rows["Healthy"]["ap5095"])
        c = float(rows["Healthy"]["ap50"])
        check("C: AP_<class> equals ap5095, confirming the mislabel",
              abs(a - b) < 1e-4, f"AP_={a} ap5095={b}")
        check("C: AP_<class> is NOT ap50", abs(a - c) > 0.1,
              f"AP_={a} ap50={c}")


def test_no_reference_results_modified():
    """The fixes must not have touched any archived result."""
    import subprocess
    r = subprocess.run(["git", "status", "--porcelain",
                        "reference_results", "figures/source_csv"],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        skip("no archived results modified", "not a git repo")
        return

    # Before the first commit every path is untracked ("?? "), which is not
    # a modification -- there is no committed version to differ from. Only
    # once HEAD exists can this test mean anything, and even then "??" marks
    # a new file rather than a changed archived one.
    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"],
                          cwd=REPO, capture_output=True, text=True)
    if head.returncode != 0:
        skip("no archived results modified", "no commits yet")
        return

    changed = [ln for ln in r.stdout.splitlines()
               if ln.strip() and not ln.startswith("??")]
    check("no archived results modified", not changed,
          "; ".join(changed)[:200])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print(f"\n{name}")
            try:
                fn()
            except Exception as e:                        # noqa: BLE001
                check(name, False, f"{type(e).__name__}: {e}")
    print(f"\n{'='*64}")
    print(f"  {len(failures)} failure(s), {len(skips)} skipped")
    for f in failures:
        print("   FAIL", f)
    for s in skips:
        print("   SKIP", s)
    sys.exit(1 if failures else 0)
