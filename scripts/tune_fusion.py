"""Measure fusion weights on YOUR labelled images, then print the env line to set.

    python scripts/tune_fusion.py --data-dir data/labelled

`--data-dir` is an ImageFolder tree with one real-ish and one fake-ish class, same convention
as train_npr.py. Every image goes through the service's own analyze(), so the signals being
tuned are exactly the ones production computes — not a simulation of them.

Output is a grid search over the weights plus balanced accuracy and AUC per candidate. It
prints numbers; it does not write them anywhere or claim them anywhere. Copy the winner into
FUSION_WEIGHTS, and the measured accuracy into app/model-card/page.tsx — by hand, having
looked at it.

This replaces guessing at fusion weights. It is not a substitute for cross-dataset evaluation:
weights tuned on one generator's output are weights for that generator.
"""

import argparse
import itertools
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image, UnidentifiedImageError

import inference_server as svc
from common.metrics import binary_auc

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fake_class_index(classes):
    """Which folder holds the fakes, by name. The service's own rules, so one vocabulary."""
    flags = [svc.label_is_fake(c) for c in classes]
    if len(classes) != 2 or True not in flags or False not in flags:
        sys.exit(f"Need exactly two folders, one real-ish and one fake-ish. Got {classes}.")
    return flags.index(True)


def collect(data_dir):
    """[(path, is_fake)] from the two class folders."""
    classes = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    fake_i = fake_class_index(classes)  # reuses the service's own label rules — torch-free
    items = []
    for i, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        for name in sorted(os.listdir(folder)):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                items.append((os.path.join(folder, name), i == fake_i))
    print(f"{len(items)} images | classes {classes} | fake = '{classes[fake_i]}'")
    return items


def measure(items, limit):
    """Run every image once and keep its raw signals. The grid search then costs nothing."""
    rows, failed = [], 0
    for n, (path, is_fake) in enumerate(items[:limit] if limit else items):
        try:
            with Image.open(path) as raw:
                out = svc.analyze(raw.convert("RGB"))
        except (UnidentifiedImageError, OSError):
            failed += 1
            continue
        s = out["signals"]
        rows.append({
            "path": path,
            "fake": is_fake,
            "face": s["modelScore"],
            "npr": s["nprScore"],
            "frequency": float(s["frequencyScore"]) if s["frequencyScore"] is not None else None,
        })
        if (n + 1) % 25 == 0:
            print(f"  ...{n + 1} scanned")
    if failed:
        print(f"  {failed} unreadable files skipped")
    return rows


def score_weights(rows, weights):
    """Balanced accuracy at the live thresholds, plus AUC over the fused score.

    Balanced, not plain, accuracy: a set that is 90% real makes "always real" look excellent.
    Abstentions (uncertain) count as wrong — an honest cost for the honest middle band.
    """
    fused, labels = [], []
    for r in rows:
        value, used = svc.fuse({k: r[k] for k in ("face", "npr", "frequency")}, weights)
        if value is None:
            continue
        fused.append(value)
        labels.append(1 if r["fake"] else 0)

    if not fused:
        return None

    correct = {0: 0, 1: 0}
    totals = {0: 0, 1: 0}
    for value, label in zip(fused, labels):
        verdict, _ = svc.verdict_for(value)
        totals[label] += 1
        if (label == 1 and verdict == "fake") or (label == 0 and verdict == "real"):
            correct[label] += 1

    recalls = [correct[c] / totals[c] for c in (0, 1) if totals[c]]
    return {
        "balanced_acc": sum(recalls) / len(recalls) if recalls else float("nan"),
        "recall_real": correct[0] / totals[0] if totals[0] else float("nan"),
        "recall_fake": correct[1] / totals[1] if totals[1] else float("nan"),
        "auc": binary_auc(fused, labels),
        "n": len(fused),
    }


def selfcheck():
    """Weights that match the informative signal must win. Uses fabricated rows on purpose —
    this checks the search, not any model."""
    rows = []
    for i in range(20):
        fake = i % 2 == 0
        rows.append({
            "fake": fake,
            "npr": 95.0 if fake else 5.0,   # informative
            "face": 50.0,                    # useless: same for both classes
            "frequency": 50.0,
        })
    npr_only = score_weights(rows, {"face": 0.0, "npr": 1.0, "frequency": 0.0})
    face_only = score_weights(rows, {"face": 1.0, "npr": 0.0, "frequency": 0.0})
    assert npr_only["balanced_acc"] == 1.0, npr_only
    assert face_only["balanced_acc"] == 0.0, "a useless signal must not look accurate"
    assert npr_only["auc"] == 1.0
    assert score_weights([{"fake": True, "face": None, "npr": None, "frequency": None}],
                         {"face": 1.0, "npr": 1.0, "frequency": 0.0}) is None
    print("tune_fusion selfcheck passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="ImageFolder: one real class, one fake class")
    ap.add_argument("--limit", type=int, default=0, help="stop after N images (smoke test)")
    ap.add_argument("--step", type=float, default=0.1, help="weight grid resolution")
    ap.add_argument("--with-frequency", action="store_true",
                    help="also search frequency weights (default: fixed at 0)")
    ap.add_argument("--json-out", default=None, help="write the measured per-image signals here")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    if not args.data_dir:
        sys.exit("--data-dir is required.")

    svc.startup()
    if svc.PREDICT is None and svc.NPR_PREDICT is None:
        sys.exit("No detector loaded — nothing to tune.")

    rows = measure(collect(args.data_dir), args.limit)
    if not rows:
        sys.exit("No readable images.")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"per-image signals -> {args.json_out}")

    steps = [round(i * args.step, 3) for i in range(int(1 / args.step) + 1)]
    freqs = steps if args.with_frequency else [0.0]
    results = []
    for face, npr, freq in itertools.product(steps, steps, freqs):
        if face + npr + freq <= 0:
            continue
        weights = {"face": face, "npr": npr, "frequency": freq}
        m = score_weights(rows, weights)
        if m:
            results.append((m["balanced_acc"], m["auc"], weights, m))

    results.sort(key=lambda r: (r[0], r[1]), reverse=True)
    print("\ntop 10 weight sets (balanced accuracy, then AUC):")
    for acc, auc, weights, m in results[:10]:
        spec = ",".join(f"{k}={v}" for k, v in weights.items() if v)
        print(f"  {spec:<34} balanced {acc:.3f}  AUC {auc:.3f}  "
              f"real {m['recall_real']:.3f} / fake {m['recall_fake']:.3f}  n={m['n']}")

    best = results[0]
    print("\ncurrent default:", ",".join(f"{k}={v}" for k, v in svc.FUSION_WEIGHTS.items()))
    print("measured best  :", ",".join(f"{k}={v}" for k, v in best[2].items()))
    print(f"\nFUSION_WEIGHTS=\"{','.join(f'{k}={v}' for k, v in best[2].items())}\"")
    print("\nThese numbers describe THIS set only. Re-measure on images from a generator the "
          "detectors have not seen before you put anything on the model card.")


if __name__ == "__main__":
    main()
