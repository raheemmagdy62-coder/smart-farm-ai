"""evaluate_plants.py - measure plant identification on YOUR OWN labeled photos.

    python evaluate_plants.py my_photos
    python evaluate_plants.py my_photos --out results.csv

Folder layout (folder name = the true label):
    my_photos/grape/  my_photos/guava/  my_photos/pear/
    my_photos/other_fruit/   (leaves of any other plant)
    my_photos/not_leaf/      (hands, soil, backgrounds, ...)

Use photos that were NEVER used to choose thresholds or to train anything.
The numbers printed here are the only honest way to know how well the system works.
The softmax score is NOT accuracy: accuracy is measured here by comparing with the true labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import plant_model as pm

FOLDER_TO_TRUE = {
    "grape": "grape", "guava": "guava", "pear": "pear",
    "other_fruit": "other", "not_leaf": "other",
}
BUCKETS = ["grape", "guava", "pear", "other"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CONFIDENCE_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]


def bucket_of(key, core):
    return key if key in core else "other"


def summarize(records):
    """Metrics from a list of per-image records (pure function, easy to test)."""
    n = len(records)
    confusion = {t: {p: 0 for p in BUCKETS} for t in BUCKETS}
    for r in records:
        confusion[r["true"]][r["pred"]] += 1

    per_class = {}
    for cls in BUCKETS:
        tp = confusion[cls][cls]
        fp = sum(confusion[t][cls] for t in BUCKETS) - tp
        fn = sum(confusion[cls].values()) - tp
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if precision is not None and recall is not None else None)
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(confusion[cls].values())}

    correct = sum(confusion[c][c] for c in BUCKETS)
    likely = [r for r in records if r["status"] == "likely"]
    likely_correct = [r for r in likely if r["true"] == r["pred"]]
    others = [r for r in records if r["true"] == "other"]
    false_accepts = [r for r in others if r["status"] == "likely"]

    calibration = []
    claimed = [r for r in records if r["pred"] != "other"]
    for lo, hi in zip(CONFIDENCE_BINS[:-1], CONFIDENCE_BINS[1:]):
        in_bin = [r for r in claimed if lo <= r["score"] < hi]
        acc = sum(1 for r in in_bin if r["true"] == r["pred"]) / len(in_bin) if in_bin else None
        calibration.append({"low": lo, "high": min(hi, 1.0), "n": len(in_bin), "accuracy": acc})

    return {
        "n": n,
        "accuracy_top1": correct / n if n else None,
        "confusion": confusion,
        "per_class": per_class,
        "coverage_likely": len(likely) / n if n else None,
        "accuracy_when_likely": len(likely_correct) / len(likely) if likely else None,
        "false_accept_rate_on_other": len(false_accepts) / len(others) if others else None,
        "n_other": len(others),
        "calibration": calibration,
    }


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_summary(s):
    print("\n================ RESULTS ================")
    print(f"images: {s['n']}   top-1 accuracy (grape/guava/pear/other): {fmt(s['accuracy_top1'])}")
    print("\nper class (bucket = what the model chose; 'other' = anything that is not grape/guava/pear):")
    for cls, m in s["per_class"].items():
        print(f"  {cls:<8} precision {fmt(m['precision'], 2)}  recall {fmt(m['recall'], 2)}  F1 {fmt(m['f1'], 2)}  (n={m['support']})")
    print("\nconfusion matrix (rows = true, columns = predicted):")
    print("          " + "".join(f"{p:>8}" for p in BUCKETS))
    for t in BUCKETS:
        print(f"  {t:<8}" + "".join(f"{s['confusion'][t][p]:>8}" for p in BUCKETS))
    print(f"\nshare of photos called 'likely' (coverage): {fmt(s['coverage_likely'])}")
    print(f"accuracy among 'likely' answers:            {fmt(s['accuracy_when_likely'])}")
    print(f"FALSE ACCEPT rate (other/not-leaf photos called 'likely' grape/guava/pear): {fmt(s['false_accept_rate_on_other'])}  (n={s['n_other']})")
    print("\ncalibration: when the model claims grape/guava/pear, how often is it right per score bin?")
    for c in s["calibration"]:
        print(f"  score {c['low']:.1f}-{c['high']:.1f}: n={c['n']:<4} accuracy {fmt(c['accuracy'], 2)}")
    print("\nREMEMBER: small test sets give wide uncertainty. Report n next to every number.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate plant identification on labeled photos.")
    parser.add_argument("folder")
    parser.add_argument("--out", default="evaluation_results.csv")
    args = parser.parse_args(argv)

    root = Path(args.folder)
    if not root.is_dir():
        sys.exit(f"Folder not found: {root}")

    catalog = pm.load_catalog()
    index = pm.species_index(catalog)
    core = set(pm.core_keys(catalog))

    print("Loading BioCLIP (first time downloads the model) ...")
    bundle = pm.load_model(catalog=catalog)

    from PIL import Image, ImageOps

    records = []
    for folder, true_bucket in FOLDER_TO_TRUE.items():
        sub = root / folder
        if not sub.is_dir():
            print(f"(folder '{folder}' not found - skipped)")
            continue
        files = sorted(p for p in sub.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        print(f"{folder}: {len(files)} images")
        for path in files:
            try:
                image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                print(f"  skipped {path.name}: {type(exc).__name__}")
                continue
            started = time.time()
            result = pm.classify(image, bundle)
            decision = pm.decide(result["ranked"], result["max_similarity"], core)
            top3 = "; ".join(f"{k}:{s:.2f}" for k, s in result["ranked"][:3])
            records.append({
                "file": str(path.relative_to(root)), "folder": folder, "true": true_bucket,
                "top_key": decision["key"], "top_name_en": pm.display_names(decision["key"], index)[1],
                "pred": bucket_of(decision["key"], core), "status": decision["status"],
                "score": decision["score"], "margin": decision["margin"],
                "max_similarity": result["max_similarity"], "top3": top3,
                "seconds": round(time.time() - started, 2),
            })

    if not records:
        sys.exit("No images were evaluated.")

    with open(args.out, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nPer-image results saved to {args.out}")
    print(f"Model: {pm.MODEL_ID}   thresholds: HIGH={pm.HIGH_CONFIDENCE} LOW={pm.LOW_CONFIDENCE} MARGIN={pm.MIN_MARGIN} RAW_SIM={pm.MIN_RAW_SIMILARITY}")
    print_summary(summarize(records))


if __name__ == "__main__":
    main()
