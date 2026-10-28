"""plant_model.py - plant identification for Smart Farm AI.

Model: BioCLIP 2 (open_clip), used ZERO-SHOT: the image is compared with text
prompts built from a species catalog (data/species_catalog.csv). Nothing here
was trained by us, and nothing here has been measured on our own photos yet.

This file has NO Streamlit code, so app.py, evaluate_plants.py and disease.py
can all use exactly the same logic. All the maths after the model call is plain
numpy, which makes it easy to test without downloading the model.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "data" / "species_catalog.csv"

# To use the smaller / faster (but weaker) model: "hf-hub:imageomics/bioclip"
MODEL_ID = "hf-hub:imageomics/bioclip-2"
MODEL_NAME = "BioCLIP 2"

# Decision thresholds. HAND-PICKED starting values, NOT measured yet.
# evaluate_plants.py measures how they behave on labeled photos.
HIGH_CONFIDENCE = 0.80   # top score needed to call a result "likely"
LOW_CONFIDENCE = 0.50    # below this the answer is "cannot decide"
MIN_MARGIN = 0.25        # top score must beat the runner-up by this much
# Optional open-set guard on the RAW cosine similarity (before softmax).
# None = disabled. Set a number ONLY after measuring it on real data.
MIN_RAW_SIMILARITY = None

NOT_A_PLANT_KEY = "not_plant"
NOT_A_PLANT_NAME_AR = "ليست ورقة نبات"
CUSTOM_PREFIX = "custom:"

NON_PLANT_PROMPTS = [
    "a photo of a person.",
    "a photo of an animal.",
    "a photo of a building.",
    "a photo of a document or a screenshot.",
    "a photo of an object.",
]


@dataclass
class Species:
    key: str
    name_ar: str
    name_en: str
    scientific: list = field(default_factory=list)
    category: str = "other"
    core: bool = False


def load_catalog(path=CATALOG_PATH):
    """Read the species catalog CSV. Add a row there to add a species."""
    species = []
    seen = set()
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("key") or "").strip()
            if not key:
                continue
            if key in seen:
                raise ValueError(f"Duplicate key in species catalog: {key}")
            seen.add(key)
            names = [n.strip() for n in (row.get("scientific") or "").split("|") if n.strip()]
            if not names:
                raise ValueError(f"Species '{key}' has no scientific name")
            species.append(
                Species(
                    key=key,
                    name_ar=(row.get("name_ar") or key).strip(),
                    name_en=(row.get("name_en") or key).strip(),
                    scientific=names,
                    category=(row.get("category") or "other").strip(),
                    core=(row.get("core") or "0").strip() == "1",
                )
            )
    return species


def core_keys(catalog):
    return [s.key for s in catalog if s.core]


def species_index(catalog):
    return {s.key: s for s in catalog}


def prompts_for(scientific_name):
    """The text descriptions the image is compared with, for one scientific name."""
    return [
        f"a photo of {scientific_name}.",
        f"a photo of the leaf of {scientific_name}.",
        scientific_name,
    ]


def build_candidates(catalog):
    """[(key, [prompts])]: one entry per scientific name + the non-plant prompts."""
    candidates = []
    for sp in catalog:
        for name in sp.scientific:
            candidates.append((sp.key, prompts_for(name)))
    for prompt in NON_PLANT_PROMPTS:
        candidates.append((NOT_A_PLANT_KEY, [prompt]))
    return candidates


def display_names(key, catalog_index):
    """(arabic, english) names for a result key."""
    if key == NOT_A_PLANT_KEY:
        return NOT_A_PLANT_NAME_AR, "Not a plant"
    if key.startswith(CUSTOM_PREFIX):
        name = key[len(CUSTOM_PREFIX):]
        return name, name
    sp = catalog_index.get(key)
    if sp is None:
        return key, key
    return sp.name_ar, sp.name_en


# ---------------------------------------------------------------
# Loading the model (needs torch + open_clip_torch). Imports are lazy.
# ---------------------------------------------------------------
def load_model(model_id=MODEL_ID, catalog=None):
    """Load BioCLIP and pre-compute the text vectors of the whole catalog.

    Returns a "bundle" dict that contains only plain functions and numpy
    arrays, so classify() below does not need torch.
    The first call downloads the model (large, one time only).
    """
    import open_clip
    import torch

    catalog = catalog if catalog is not None else load_catalog()
    model, _, preprocess = open_clip.create_model_and_transforms(model_id)
    tokenizer = open_clip.get_tokenizer(model_id)
    model.eval()

    def encode_image(image):
        with torch.inference_mode():
            feats = model.encode_image(preprocess(image).unsqueeze(0))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].float().cpu().numpy()

    def encode_text(prompts):
        with torch.inference_mode():
            feats = model.encode_text(tokenizer(list(prompts)))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

    keys, vectors = [], []
    for key, prompts in build_candidates(catalog):
        mean = encode_text(prompts).mean(axis=0)
        vectors.append(mean / np.linalg.norm(mean))
        keys.append(key)

    return {
        "model_id": model_id,
        "catalog": catalog,
        "encode_image": encode_image,
        "encode_text": encode_text,
        "text_matrix": np.stack(vectors).astype(np.float32),
        "keys": keys,
        "logit_scale": float(model.logit_scale.exp().item()),
    }


# ---------------------------------------------------------------
# Scoring (pure numpy)
# ---------------------------------------------------------------
def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def classify(image, bundle, extra_species=None):
    """Compare one PIL image with every candidate.

    extra_species: optional list of extra scientific names typed by the user.
    They are added as candidates for this call only (no code change needed).

    Returns a dict:
      ranked          [(key, score)] best first. Scores add up to 1 over the
                      candidates; this is the model's PREFERENCE, not accuracy.
      max_similarity  highest raw cosine similarity (before softmax)
      embedding       the normalized image vector (reused by disease.py)
    """
    embedding = bundle["encode_image"](image)
    text_matrix = bundle["text_matrix"]
    keys = list(bundle["keys"])

    extra_vectors = []
    for name in extra_species or []:
        name = name.strip()
        if not name:
            continue
        mean = bundle["encode_text"](prompts_for(name)).mean(axis=0)
        extra_vectors.append(mean / np.linalg.norm(mean))
        keys.append(CUSTOM_PREFIX + name)
    if extra_vectors:
        text_matrix = np.vstack([text_matrix, np.stack(extra_vectors)])

    similarities = text_matrix @ embedding
    scores = softmax(bundle["logit_scale"] * similarities)

    totals = {}
    for key, score in zip(keys, scores):
        totals[key] = totals.get(key, 0.0) + float(score)
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return {
        "ranked": ranked,
        "max_similarity": float(similarities.max()),
        "embedding": embedding,
    }


# ---------------------------------------------------------------
# Honest decision rules
# ---------------------------------------------------------------
def decide(ranked, max_similarity=None, core=()):
    """Turn scores into a status. The status is a CAUTIOUS label, not a diagnosis.

    status:
      likely     top candidate is one of our core plants, clear and confident
      uncertain  closest to a core plant but not confident enough
      other      closest to a plant that is not one of our core plants
      not_plant  closest to a "not a plant" prompt
      unknown    no candidate is convincing (or the raw similarity is too low)
    """
    top_key, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - second_score

    if MIN_RAW_SIMILARITY is not None and max_similarity is not None and max_similarity < MIN_RAW_SIMILARITY:
        status = "unknown"
    elif top_score < LOW_CONFIDENCE:
        status = "unknown"
    elif top_key == NOT_A_PLANT_KEY:
        status = "not_plant"
    elif top_key not in core:
        status = "other"
    elif top_score >= HIGH_CONFIDENCE and margin >= MIN_MARGIN:
        status = "likely"
    else:
        status = "uncertain"

    return {"status": status, "key": top_key, "score": top_score, "margin": margin}


def image_quality_notes(image, min_side=100):
    """Cheap checks on the photo. Returns a list of Arabic warnings (may be empty)."""
    from PIL import ImageStat

    notes = []
    if min(image.size) < min_side:
        notes.append("الصورة صغيرة جدًا، وقد لا تكون النتيجة موثوقة.")
    mean_brightness = ImageStat.Stat(image.convert("L")).mean[0]
    if mean_brightness < 40:
        notes.append("الصورة تبدو مظلمة جدًا. جرّب إضاءة أفضل.")
    elif mean_brightness > 225:
        notes.append("الصورة تبدو ساطعة جدًا. جرّب تقليل الإضاءة المباشرة.")
    return notes
