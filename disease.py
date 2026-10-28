"""disease.py - "possible symptoms" heads on top of frozen BioCLIP image vectors.

IDEA (transfer learning, the cheap and honest version):
  1. BioCLIP turns each leaf photo into a vector (we do NOT retrain BioCLIP).
  2. We train a small classifier (softmax regression, numpy only) on vectors of
     LABELED photos of ONE plant species: healthy + each disease class.
  3. We measure it on photos it never saw (a separate test split) and save the
     numbers next to the model. The app shows those numbers to the user.

Nothing is shipped in models/ by default. Until you train a head for a species,
the app says that no validated disease model exists for it.

TRAIN (PowerShell, project folder, venv active):
    python disease.py train --species grape --data data_disease/grape

Folder layout (one folder per class, the folder name is the class name):
    data_disease/grape/healthy/*.jpg
    data_disease/grape/black_rot/*.jpg
    data_disease/grape/downy_mildew/*.jpg

IMPORTANT about leakage: if your dataset contains several copies/augmentations
of the same photo, give --group-regex so copies stay in the SAME split, e.g.
    --group-regex "^(.*?)(?:_aug\\d+)?\\.(?:jpg|jpeg|png)$"
Use ORIGINAL photos only whenever possible.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
HEALTHY_NAMES = {"healthy", "سليم", "سليمة"}
MIN_PROBABILITY = 0.70   # below this the head abstains (uncalibrated, hand-picked)
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)


# ---------------------------------------------------------------
# Small numpy "machine learning" helpers
# ---------------------------------------------------------------
def _softmax_rows(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_softmax_regression(x, y, n_classes, l2=1e-3, lr=0.05, epochs=300, balance=True):
    """Multinomial logistic regression trained with full-batch Adam."""
    n, d = x.shape
    w = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    onehot = np.eye(n_classes)[y]
    if balance:
        counts = np.bincount(y, minlength=n_classes)
        sample_w = (n / (n_classes * np.maximum(counts, 1)))[y]
    else:
        sample_w = np.ones(n)
    sample_w = sample_w / sample_w.sum()

    mw, vw = np.zeros_like(w), np.zeros_like(w)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for step in range(1, epochs + 1):
        probs = _softmax_rows(x @ w + b)
        grad_logits = (probs - onehot) * sample_w[:, None]
        gw = x.T @ grad_logits + l2 * w
        gb = grad_logits.sum(axis=0)
        mw = beta1 * mw + (1 - beta1) * gw
        vw = beta2 * vw + (1 - beta2) * gw ** 2
        mb = beta1 * mb + (1 - beta1) * gb
        vb = beta2 * vb + (1 - beta2) * gb ** 2
        w -= lr * (mw / (1 - beta1 ** step)) / (np.sqrt(vw / (1 - beta2 ** step)) + eps)
        b -= lr * (mb / (1 - beta1 ** step)) / (np.sqrt(vb / (1 - beta2 ** step)) + eps)
    return w, b


def classification_report(y_true, y_pred, class_names):
    """Accuracy, per-class precision/recall/F1, macro F1 and the confusion matrix."""
    k = len(class_names)
    confusion = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    per_class = {}
    f1_values = []
    for i, name in enumerate(class_names):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(confusion[i, :].sum())}
        f1_values.append(f1)
    total = confusion.sum()
    return {
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion": confusion.tolist(),
        "n": int(total),
    }


def abstention_curve(probs, y_true, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """For each threshold: how many photos get an answer, and how accurate those answers are."""
    top = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    rows = []
    for t in thresholds:
        answered = conf >= t
        n_answered = int(answered.sum())
        acc = float((top[answered] == y_true[answered]).mean()) if n_answered else None
        rows.append({"threshold": t, "coverage": n_answered / len(y_true) if len(y_true) else 0.0,
                     "accuracy_when_answered": acc, "n_answered": n_answered})
    return rows


def split_indices(labels, groups, seed=0, fractions=(0.6, 0.2, 0.2)):
    """Group-aware, per-class split into train / val / test index lists.

    All photos that share a group id (copies of one photo) stay in one split.
    With one group per photo this is a plain stratified random split.
    """
    rng = np.random.default_rng(seed)
    splits = ([], [], [])
    labels = np.asarray(labels)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        group_ids = sorted({groups[i] for i in idx})
        rng.shuffle(group_ids)
        n_groups = len(group_ids)
        n_val = max(1, round(n_groups * fractions[1])) if n_groups >= 3 else 0
        n_test = max(1, round(n_groups * fractions[2])) if n_groups >= 3 else 0
        val_groups = set(group_ids[:n_val])
        test_groups = set(group_ids[n_val:n_val + n_test])
        for i in idx:
            g = groups[i]
            if g in val_groups:
                splits[1].append(i)
            elif g in test_groups:
                splits[2].append(i)
            else:
                splits[0].append(i)
    return [np.array(sorted(s), dtype=int) for s in splits]


def fit_and_evaluate(features, labels, class_names, groups=None, seed=0):
    """Choose L2 on the validation split, then report ONCE on the test split."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    groups = groups if groups is not None else list(range(len(labels)))
    train_i, val_i, test_i = split_indices(labels, groups, seed)
    if len(val_i) == 0 or len(test_i) == 0:
        raise ValueError("Not enough photos per class for a train/validation/test split (need at least 3 groups per class).")

    mean = features[train_i].mean(axis=0)
    std = features[train_i].std(axis=0) + 1e-6
    scaled = (features - mean) / std
    k = len(class_names)

    best = None
    for l2 in L2_GRID:
        w, b = train_softmax_regression(scaled[train_i], labels[train_i], k, l2=l2)
        pred = (scaled[val_i] @ w + b).argmax(axis=1)
        val_report = classification_report(labels[val_i], pred, class_names)
        if best is None or val_report["macro_f1"] > best["val_macro_f1"]:
            best = {"l2": l2, "w": w, "b": b, "val_macro_f1": val_report["macro_f1"]}

    test_probs = _softmax_rows(scaled[test_i] @ best["w"] + best["b"])
    test_report = classification_report(labels[test_i], test_probs.argmax(axis=1), class_names)
    return {
        "w": best["w"], "b": best["b"], "mean": mean, "std": std, "l2": best["l2"],
        "val_macro_f1": best["val_macro_f1"],
        "test_report": test_report,
        "abstention": abstention_curve(test_probs, labels[test_i]),
        "sizes": {"train": len(train_i), "val": len(val_i), "test": len(test_i)},
        "group_split": groups is not None and len(set(groups)) < len(labels),
    }


# ---------------------------------------------------------------
# Using a trained head
# ---------------------------------------------------------------
class DiseaseHead:
    def __init__(self, w, b, mean, std, classes, metrics=None):
        self.w, self.b, self.mean, self.std = w, b, mean, std
        self.classes = list(classes)
        self.metrics = metrics or {}

    @classmethod
    def load(cls, species_key, models_dir=MODELS_DIR):
        npz_path = Path(models_dir) / f"disease_{species_key}.npz"
        if not npz_path.exists():
            return None
        data = np.load(npz_path, allow_pickle=False)
        classes = json.loads(str(data["classes"]))
        metrics_path = Path(models_dir) / f"disease_{species_key}.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        return cls(data["w"], data["b"], data["mean"], data["std"], classes, metrics)

    def predict(self, embedding):
        x = (np.asarray(embedding, dtype=np.float64) - self.mean) / self.std
        probs = _softmax_rows((x @ self.w + self.b)[None, :])[0]
        return sorted(zip(self.classes, probs.tolist()), key=lambda item: item[1], reverse=True)

    def interpret(self, embedding, min_probability=MIN_PROBABILITY):
        """status: healthy | possible_symptoms | uncertain. Never a final diagnosis."""
        ranked = self.predict(embedding)
        top_name, top_prob = ranked[0]
        if top_prob < min_probability:
            status = "uncertain"
        elif top_name.lower() in HEALTHY_NAMES:
            status = "healthy"
        else:
            status = "possible_symptoms"
        return {"status": status, "top": top_name, "probability": top_prob, "ranked": ranked}


def save_head(species_key, result, class_names, extra, models_dir=MODELS_DIR):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        models_dir / f"disease_{species_key}.npz",
        w=result["w"], b=result["b"], mean=result["mean"], std=result["std"],
        classes=np.array(json.dumps(list(class_names), ensure_ascii=False)),
    )
    metrics = {
        "species": species_key,
        "classes": list(class_names),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "l2": result["l2"],
        "val_macro_f1": result["val_macro_f1"],
        "sizes": result["sizes"],
        "test": result["test_report"],
        "abstention_on_test": result["abstention"],
        "group_split": result["group_split"],
        **extra,
    }
    (models_dir / f"disease_{species_key}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


# ---------------------------------------------------------------
# Command line: python disease.py train ...
# ---------------------------------------------------------------
def collect_images(data_dir):
    data_dir = Path(data_dir)
    class_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    items = []
    for ci, name in enumerate(class_names):
        for path in sorted((data_dir / name).rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                items.append((path, ci))
    return class_names, items


def extract_embeddings(items, encode_image):
    from PIL import Image, ImageOps

    vectors, labels, kept = [], [], []
    for number, (path, label) in enumerate(items, 1):
        try:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            vectors.append(encode_image(image))
            labels.append(label)
            kept.append(path)
        except Exception as exc:  # noqa: BLE001 - skip unreadable files, keep going
            print(f"  skipped {path.name}: {type(exc).__name__}")
        if number % 50 == 0:
            print(f"  {number}/{len(items)} images")
    return np.array(vectors), np.array(labels), kept


def run_train(args):
    import plant_model

    class_names, items = collect_images(args.data)
    if len(class_names) < 2:
        sys.exit("Need at least 2 class folders (for example healthy and one disease).")
    print(f"Classes: {class_names}; images found: {len(items)}")

    cache = MODELS_DIR / f"_cache_{args.species}.npz"
    MODELS_DIR.mkdir(exist_ok=True)
    if cache.exists() and not args.recompute:
        data = np.load(cache, allow_pickle=False)
        features, labels, names = data["features"], data["labels"], list(data["names"])
        print(f"Loaded cached image vectors: {cache.name} (use --recompute to rebuild)")
    else:
        print("Loading BioCLIP (first time downloads the model) ...")
        bundle = plant_model.load_model()
        features, labels, kept = extract_embeddings(items, bundle["encode_image"])
        names = [p.name for p in kept]
        np.savez(cache, features=features, labels=labels, names=np.array(names))

    groups = None
    if args.group_regex:
        pattern = re.compile(args.group_regex)
        groups = []
        for name, label in zip(names, labels):
            m = pattern.match(name)
            groups.append(f"{label}:{m.group(1) if m else name}")
    else:
        print("WARNING: no --group-regex given, so copies of the same photo may land in different splits and inflate the test numbers.")

    result = fit_and_evaluate(features, labels, class_names, groups, seed=args.seed)
    metrics = save_head(args.species, result, class_names, {"model_id": plant_model.MODEL_ID, "seed": args.seed})

    test = metrics["test"]
    print("\n=== TEST SET (photos the model never saw) ===")
    print(f"photos: {test['n']}   accuracy: {test['accuracy']:.3f}   macro F1: {test['macro_f1']:.3f}")
    for name, m in test["per_class"].items():
        print(f"  {name:<24} precision {m['precision']:.2f}  recall {m['recall']:.2f}  F1 {m['f1']:.2f}  (n={m['support']})")
    print("confusion matrix (rows = true, columns = predicted):")
    for name, row in zip(class_names, test["confusion"]):
        print(f"  {name:<24} {row}")
    print("abstention on test (threshold -> coverage, accuracy when answered):")
    for r in metrics["abstention_on_test"]:
        acc = "n/a" if r["accuracy_when_answered"] is None else f"{r['accuracy_when_answered']:.3f}"
        print(f"  {r['threshold']:.1f} -> coverage {r['coverage']:.2f}, accuracy {acc}")
    print(f"\nSaved: models/disease_{args.species}.npz and .json")
    print("These numbers describe THIS dataset only. Test again on your own field photos before trusting it.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a disease-symptom head for one plant species.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--species", required=True, help="species key, e.g. grape")
    train.add_argument("--data", required=True, help="folder with one sub-folder per class")
    train.add_argument("--group-regex", default=None, help="regex whose group(1) is the original-photo id")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--recompute", action="store_true", help="ignore the cached image vectors")
    args = parser.parse_args(argv)
    if args.command == "train":
        run_train(args)


if __name__ == "__main__":
    main()
