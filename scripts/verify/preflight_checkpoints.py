"""
preflight_checkpoints.py
────────────────────────
Run BEFORE any Phase D.2 GPU evaluation.

Confirms every checkpoint the evaluation needs is present and that its
SHA-256 matches `manifests/checkpoint_inventory.csv`. A missing file or a
hash mismatch is reported and the exit code is non-zero; nothing is
substituted and no evaluation is started.

The specific hazard this guards against: two checkpoints exist for
S3 (retinanet_convnext_fpn_cbam) seed 456 and they produce results 3.45 pp
apart. The one in `outputs/checkpoints/` is the SEPTEMBER file, which has no
training log. The canonical one is the JULY file in
`outputs/checkpoints_recovered/`. Both are listed here so the difference is
visible rather than silently resolved.

Usage
    python scripts/verify/preflight_checkpoints.py
    python scripts/verify/preflight_checkpoints.py --quick     # size only
    python scripts/verify/preflight_checkpoints.py --json reports/preflight.json

Hashing 7.86 GB takes a few minutes. --quick checks existence and size only,
which catches a missing or truncated download but not a wrong file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
while not os.path.exists(os.path.join(REPO, "train_all.py")) \
        and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
INVENTORY = os.path.join(REPO, "manifests", "checkpoint_inventory.csv")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="check existence and size only, skip hashing")
    ap.add_argument("--json", default=None)
    ap.add_argument("--root", default=REPO,
                    help="where outputs/checkpoints/ lives, if not the repo")
    args = ap.parse_args()

    if not os.path.exists(INVENTORY):
        sys.exit(f"inventory not found: {INVENTORY}")
    rows = list(csv.DictReader(open(INVENTORY, encoding="utf-8")))

    results = []
    print(f"preflight: {len(rows)} checkpoints in the inventory")
    print(f"mode     : {'size only (--quick)' if args.quick else 'full SHA-256'}")
    print(f"root     : {args.root}\n")
    print(f"{'stage':<36}{'seed':>5}{'req':>5}  {'status'}")
    print("-" * 78)

    for r in rows:
        path = os.path.join(args.root, r["path"])
        required = r["required_for_paper"].startswith("YES")
        rec = dict(stage=r["stage"], seed=r["seed"], path=r["path"],
                   required=required, expected_sha256=r["sha256"],
                   expected_size=int(r["size_bytes"]))

        if not os.path.exists(path):
            rec["status"] = "MISSING" if required else "absent (not required)"
            rec["actual_sha256"] = None
        else:
            size = os.path.getsize(path)
            rec["actual_size"] = size
            if size != int(r["size_bytes"]):
                rec["status"] = "SIZE MISMATCH"
                rec["actual_sha256"] = None
            elif args.quick:
                rec["status"] = "size OK (not hashed)"
                rec["actual_sha256"] = None
            else:
                d = sha256(path)
                rec["actual_sha256"] = d
                rec["status"] = "OK" if d == r["sha256"] else "HASH MISMATCH"

        results.append(rec)
        note = ""
        if r["note"]:
            note = "  <- " + r["note"].split(".")[0]
        print(f"{r['stage']:<36}{r['seed']:>5}{'yes' if required else 'no':>5}"
              f"  {rec['status']}{note}")

    req = [r for r in results if r["required"]]
    bad = [r for r in req if r["status"] not in
           ("OK", "size OK (not hashed)")]

    print(f"\n{'='*78}")
    print(f"  required checkpoints : {len(req)}")
    print(f"  problems             : {len(bad)}")
    if bad:
        print("\n  EVALUATION MUST NOT PROCEED for:")
        for r in bad:
            print(f"    {r['stage']} seed {r['seed']}: {r['status']}")
            print(f"      {r['path']}")
            if r.get("actual_sha256"):
                print(f"      expected {r['expected_sha256'][:24]}...")
                print(f"      found    {r['actual_sha256'][:24]}...")
        print("\n  Do not substitute another checkpoint. Obtain the correct "
              "file or report the gap.")
    else:
        print("  all required checkpoints present and verified")

    # the S3 situation, always reported
    s3 = [r for r in results
          if "S3" in r["stage"] and r["seed"] == "456"]
    if len(s3) == 2:
        print("\n  S3 seed 456 — two checkpoints, as expected:")
        for r in s3:
            tag = "JULY, canonical" if r["expected_sha256"].startswith("0e9f3dcf") \
                  else "SEPTEMBER, disputed"
            print(f"    {tag:<22} {r['expected_sha256'][:16]}  {r['status']}")
        print("    Evaluate the July one with "
              "ablation_fpn_cbam/eval_recovered_seed456.py, which verifies "
              "the hash itself before loading.")
    print("=" * 78)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump(dict(generated=datetime.now(timezone.utc).isoformat(),
                       mode="quick" if args.quick else "sha256",
                       root=args.root, results=results),
                  open(args.json, "w", encoding="utf-8"), indent=2)
        print(f"wrote {args.json}")

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
