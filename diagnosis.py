"""diagnosis.py - guided symptom check (question and options) for Smart Farm AI.

HOW IT WORKS
  The user answers multiple-choice questions about what they SEE. Each disease in
  data/disease_kb.json lists which answers support it ("strong" / "weak") or
  contradict it ("contra"). Every such statement comes from an extension source
  listed next to the disease.

  The engine counts matching evidence. It does NOT compute a probability and it
  never states a final diagnosis. The result is a ranked list of POSSIBLE
  conditions, what is still unknown, and which question would help most next.

WHAT IT CANNOT DO
  * It only knows the conditions that are written in the knowledge base.
  * Its quality depends on the user's observations and on the sources.
  * It has not been tested against real cases yet (see evaluate_diagnosis notes in README).
"""

from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "data" / "disease_kb.json"

STRONG_POINTS = 1.0
WEAK_POINTS = 0.5
CONTRA_PENALTY = 1.5
MIN_ANSWERS = 3          # fewer informative answers than this: "insufficient"
MIN_EVIDENCE = 1.0       # best candidate must reach this to be listed as possible
AMBIGUITY_GAP = 1.0      # top and second closer than this: "ambiguous"
UNKNOWN = "unknown"
NONE = "none"


def load_kb(path=KB_PATH):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def applicable_questions(kb, plant):
    return [q for q in kb["questions"] if plant in q["applies_to"]]


def clean_answers(answers):
    """{qid: [option ids]} without empty values and without 'unknown'."""
    cleaned = {}
    for qid, value in (answers or {}).items():
        values = [value] if isinstance(value, str) else list(value or [])
        values = [v for v in values if v and v != UNKNOWN]
        if values:
            cleaned[qid] = values
    return cleaned


def informative_count(answers):
    """How many answered questions say something other than 'nothing seen'."""
    return sum(1 for values in answers.values() if any(v != NONE for v in values))


def score_diseases(kb, plant, answers):
    """Rank the diseases of one plant by matching evidence (best first)."""
    answers = clean_answers(answers)
    results = []
    for disease in kb["diseases"]:
        if disease["plant"] != plant:
            continue
        matched, contra_hits, supports, contradicts, unseen = 0.0, 0.0, [], [], []
        for qid, rule in disease["expects"].items():
            given = answers.get(qid)
            if given is None:
                if rule.get("strong"):
                    unseen.append(qid)
                continue
            strong = set(rule.get("strong", []))
            weak = set(rule.get("weak", []))
            contra = set(rule.get("contra", []))
            if strong & set(given):
                matched += STRONG_POINTS
                supports.append((qid, sorted(strong & set(given))))
            elif weak & set(given):
                matched += WEAK_POINTS
                supports.append((qid, sorted(weak & set(given))))
            if contra & set(given):
                contra_hits += 1
                contradicts.append((qid, sorted(contra & set(given))))
        results.append(
            {
                "id": disease["id"],
                "evidence": matched - CONTRA_PENALTY * contra_hits,
                "matched_points": matched,
                "supports": supports,
                "contradicts": contradicts,
                "not_yet_observed": unseen,
                "disease": disease,
            }
        )
    results.sort(key=lambda r: (-r["evidence"], -r["matched_points"], r["id"]))
    return results


def interpret(ranked, answers):
    """status: insufficient | no_symptoms | no_match | ambiguous | possible."""
    answers = clean_answers(answers)
    if len(answers) < MIN_ANSWERS:
        return {"status": "insufficient", "top": None}
    if informative_count(answers) == 0:
        return {"status": "no_symptoms", "top": None}
    if not ranked or ranked[0]["evidence"] < MIN_EVIDENCE:
        return {"status": "no_match", "top": None}
    top = ranked[0]
    close = [r for r in ranked[1:] if r["evidence"] >= MIN_EVIDENCE and top["evidence"] - r["evidence"] < AMBIGUITY_GAP]
    if close:
        return {"status": "ambiguous", "top": top, "close": close}
    return {"status": "possible", "top": top, "close": []}


def suggest_next_question(kb, plant, answers, top_n=3):
    """The unanswered question that best separates the leading candidates.

    A question scores one point for every pair of leading candidates whose
    expected answers for that question differ. Returns (question, score) or None.
    """
    answered = set(clean_answers(answers))
    ranked = score_diseases(kb, plant, answers)
    leaders = [r["disease"] for r in ranked[:top_n]]
    if len(leaders) < 2:
        return None
    best = None
    for question in applicable_questions(kb, plant):
        if question["id"] in answered:
            continue
        signatures = []
        for disease in leaders:
            rule = disease["expects"].get(question["id"], {})
            signatures.append(frozenset(rule.get("strong", []) + rule.get("weak", [])))
        score = sum(1 for i in range(len(signatures)) for j in range(i + 1, len(signatures)) if signatures[i] != signatures[j])
        if score > 0 and (best is None or score > best[1]):
            best = (question, score)
    return best


def option_text(kb, qid, option_id):
    for question in kb["questions"]:
        if question["id"] == qid:
            for option in question["options"]:
                if option["id"] == option_id:
                    return option["text_ar"]
    return option_id


def question_text(kb, qid):
    for question in kb["questions"]:
        if question["id"] == qid:
            return question["text_ar"]
    return qid
