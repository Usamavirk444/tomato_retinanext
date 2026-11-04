"""
Repository integrity tests. No GPU, no dataset, no network.

    python -m pytest tests/ -v
    python tests/test_repo_integrity.py      # also runs standalone
"""
import ast, csv, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(f"{name}: {detail}")


def test_python_syntax():
    """Every .py file parses."""
    bad = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "data")]
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                try:
                    ast.parse(open(p, encoding="utf-8").read())
                except SyntaxError as e:
                    bad.append(f"{os.path.relpath(p, REPO)}:{e.lineno}")
    check("all python files parse", not bad, "; ".join(bad[:5]))
    assert not bad, bad


def test_required_files():
    """Files the README promises actually exist."""
    required = [
        "train_all.py", "evaluate.py",
        "configs/experiment.yaml", "configs/experiment_cleantest.yaml",
        "src/models/retinanet.py", "src/models/ssdlite.py",
        "src/models/hybrid/retinanet_hybrid.py",
        "src/models/hybrid/bifpn.py", "src/models/hybrid/cbam.py",
        "src/models/hybrid/convnext_backbone.py",
        "src/training/trainer.py", "src/evaluation/evaluator.py",
        "src/dataset/yolo_dataset.py", "src/dataset/transforms.py",
        "cross_eval/cross_dataset_3_eval.py",
        "scripts/data/prepare_nine_class.py",
        "scripts/data/build_clean_testset.py",
        "scripts/data/build_clean_crossdataset.py",
        "scripts/eval/recompute_external_clean.py",
        "scripts/eval/confusion_matrix.py",
        "scripts/eval/build_results_table.py",
        "scripts/eval/benchmark_fps.py",
        "scripts/verify/validate_against_paper.py",
        "scripts/verify/verify_box_loss.py",
        "manifests/split_manifest_clean_yolo_dataset.csv",
        "manifests/checkpoint_inventory.csv",
        "figures/R/run_all.R", "figures/R/00_setup.R",
        "README.md", "QUICKSTART.md", ".gitignore",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(REPO, f))]
    check(f"{len(required)} required files present", not missing, "; ".join(missing))
    assert not missing, missing


def test_split_manifest():
    """Manifest matches the published split."""
    p = os.path.join(REPO, "manifests", "split_manifest_clean_yolo_dataset.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    from collections import Counter
    c = Counter(r["split"] for r in rows)
    check("manifest 24,876 rows", len(rows) == 24876, str(len(rows)))
    check("train 17,419", c["train"] == 17419, str(c["train"]))
    check("val 4,970", c["val"] == 4970, str(c["val"]))
    check("test 2,487", c["test"] == 2487, str(c["test"]))
    b = sum(int(r["n_boxes"]) for r in rows)
    check("67,595 boxes", b == 67595, str(b))
    assert not failures


def test_overlap_manifests():
    m = os.path.join(REPO, "manifests")
    a = {l.strip() for l in open(f"{m}/testsubset_1933_overlap_excluded.txt", encoding="utf-8") if l.strip()}
    b = {l.strip() for l in open(f"{m}/testsubset_554_overlap_leaked.txt", encoding="utf-8") if l.strip()}
    check("1,933 overlap-excluded", len(a) == 1933, str(len(a)))
    check("554 leaked", len(b) == 554, str(len(b)))
    check("disjoint, union 2,487", not (a & b) and len(a | b) == 2487)


def test_configs_load():
    import yaml
    for c in ("experiment.yaml", "experiment_cleantest.yaml"):
        d = yaml.safe_load(open(os.path.join(REPO, "configs", c), encoding="utf-8"))
        check(f"{c} loads with models+training+data", all(k in d for k in ("training", "models", "data")))


def test_learning_rate_encoded():
    """The repository must encode 1e-7, not the manuscript's 1e-6."""
    src = open(os.path.join(REPO, "train_all.py"), encoding="utf-8").read()
    n = src.count("eta_min=1e-7")
    check("eta_min=1e-7 appears 3x in train_all.py", n == 3, str(n))


def test_no_excluded_artifacts():
    bad = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            if f == ".DS_Store" or f.endswith((".pyc", ".pth", ".pt")):
                bad.append(os.path.relpath(os.path.join(root, f), REPO))
        if "__MACOSX" in dirs or "__pycache__" in dirs:
            bad.append(os.path.relpath(root, REPO) + "/<cache dir>")
    check("no checkpoints / caches / OS metadata committed", not bad, "; ".join(bad[:5]))


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        try:
            fn()
        except AssertionError:
            pass
    print(f"\n{'='*60}\n  {len(failures)} failure(s)")
    for f in failures:
        print("   -", f)
    sys.exit(1 if failures else 0)
