"""Tests for the symptom-question engine and the knowledge base file.
They check that the engine follows the KB faithfully. They are NOT proof that
the KB gives correct diagnoses for real plants."""
import re, sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import diagnosis as dg


class KnowledgeBaseIntegrity(unittest.TestCase):
    def setUp(self):
        self.kb = dg.load_kb()

    def test_references_exist(self):
        qs = {q["id"]: {o["id"] for o in q["options"]} for q in self.kb["questions"]}
        plants = set(self.kb["plants"])
        for q in self.kb["questions"]:
            self.assertIn(dg.UNKNOWN, qs[q["id"]] if q["type"] == "single" else {dg.UNKNOWN})
            self.assertTrue(set(q["applies_to"]) <= plants)
        ids = set()
        for d in self.kb["diseases"]:
            self.assertNotIn(d["id"], ids); ids.add(d["id"])
            self.assertIn(d["plant"], plants)
            for qid, rule in d["expects"].items():
                self.assertIn(qid, qs, d["id"])
                self.assertIn(d["plant"], next(q for q in self.kb["questions"] if q["id"] == qid)["applies_to"], (d["id"], qid))
                for bucket in ("strong", "weak", "contra"):
                    for oid in rule.get(bucket, []):
                        self.assertIn(oid, qs[qid], (d["id"], qid, oid))
            self.assertTrue(d["sources"], d["id"])
            for s in d["sources"]:
                self.assertTrue(s["url"].startswith("https://"), d["id"])
            self.assertTrue(d["management"], d["id"])

    def test_every_advice_has_a_source_or_is_marked_none(self):
        for d in self.kb["diseases"]:
            for item in d["management"]:
                if item["kind"] == "none":
                    self.assertEqual(item["url"], "")
                else:
                    self.assertTrue(item["url"].startswith("https://"), (d["id"], item["text_ar"][:30]))
                    self.assertTrue(item["source"], d["id"])

    def test_no_pesticide_doses_or_brands(self):
        pattern = re.compile(r"\d+(\.\d+)?\s*(مل|ملي|جم|جرام|لتر|ml|mg|g/|kg|كجم|ppm)", re.I)
        for d in self.kb["diseases"]:
            for item in d["management"]:
                self.assertIsNone(pattern.search(item["text_ar"]), (d["id"], item["text_ar"]))

    def test_each_plant_has_diseases_and_questions(self):
        for plant in self.kb["plants"]:
            self.assertGreaterEqual(len(dg.score_diseases(self.kb, plant, {})), 3)
            self.assertGreaterEqual(len(dg.applicable_questions(self.kb, plant)), 7)
        self.assertEqual(len(self.kb["diseases"]), 12)
        self.assertFalse(self.kb["reviewed_by_expert"])


class EngineBehaviour(unittest.TestCase):
    def setUp(self):
        self.kb = dg.load_kb()

    def run_case(self, plant, answers):
        ranked = dg.score_diseases(self.kb, plant, answers)
        return ranked, dg.interpret(ranked, answers)

    def test_downy_mildew_pattern(self):
        ranked, out = self.run_case("grape", {"part": ["leaf"], "leaf_top": ["yellow_spots"], "leaf_under": ["white_cottony"]})
        self.assertEqual(out["status"], "possible"); self.assertEqual(out["top"]["id"], "grape_downy_mildew")

    def test_powdery_mildew_pattern(self):
        ranked, out = self.run_case("grape", {"part": ["leaf", "fruit"], "leaf_top": ["white_powder"], "leaf_under": ["white_powder"]})
        self.assertEqual(out["top"]["id"], "grape_powdery_mildew"); self.assertEqual(out["status"], "possible")
        dm = next(r for r in ranked if r["id"] == "grape_downy_mildew")
        self.assertTrue(dm["contradicts"])

    def test_fire_blight_pattern(self):
        _, out = self.run_case("pear", {"part": ["shoot", "flower"], "shoot_look": ["scorched_crook"], "leaf_top": ["scorched_leaf"]})
        self.assertEqual(out["top"]["id"], "pear_fire_blight")

    def test_guava_anthracnose_and_wilt(self):
        _, out = self.run_case("guava", {"part": ["fruit"], "fruit_look": ["black_lesions_spores"], "leaf_top": ["brown_spots"]})
        self.assertEqual(out["top"]["id"], "guava_anthracnose")
        _, out = self.run_case("guava", {"part": ["whole_tree"], "whole_look": ["wilt_yellow_defoliation"], "waterlogging": ["waterlogged"]})
        self.assertEqual(out["top"]["id"], "guava_wilt")

    def test_guava_algal_vs_leaf_spot(self):
        _, out = self.run_case("guava", {"part": ["leaf"], "leaf_top": ["orange_tufts"], "leaf_under": ["orange_tufts"]})
        self.assertEqual(out["top"]["id"], "guava_algal_leaf_spot")
        _, out = self.run_case("guava", {"part": ["leaf"], "leaf_top": ["yellow_spots"], "leaf_under": ["brown_patches"], "leaf_age": ["old_or_lower_leaves"]})
        self.assertEqual(out["top"]["id"], "guava_leaf_spot")

    def test_similar_pear_leaf_spots_are_reported_as_ambiguous(self):
        ranked, out = self.run_case("pear", {"part": ["leaf"], "leaf_top": ["brown_spots"], "leaf_age": ["all_leaves"]})
        self.assertEqual(out["status"], "ambiguous")
        ids = {out["top"]["id"]} | {r["id"] for r in out["close"]}
        self.assertTrue({"pear_scab", "pear_fabraea_leaf_spot"} <= ids)

    def test_purple_margin_separates_fabraea(self):
        _, out = self.run_case("pear", {"part": ["leaf"], "leaf_top": ["brown_spots"], "spot_look": ["purple_margin"]})
        self.assertEqual(out["top"]["id"], "pear_fabraea_leaf_spot")

    def test_insufficient_and_unknown_answers(self):
        _, out = self.run_case("grape", {"part": ["leaf"]})
        self.assertEqual(out["status"], "insufficient")
        _, out = self.run_case("grape", {"part": ["leaf"], "leaf_top": ["unknown"], "leaf_under": ["unknown"], "spot_look": ["unknown"]})
        self.assertEqual(out["status"], "insufficient")

    def test_no_symptoms_and_no_match(self):
        _, out = self.run_case("grape", {"leaf_top": ["none"], "leaf_under": ["none"], "shoot_look": ["none"], "fruit_look": ["none"]})
        self.assertEqual(out["status"], "no_symptoms")
        _, out = self.run_case("grape", {"leaf_top": ["yellowing_all"], "shoot_look": ["oozing"], "spot_look": ["angular"]})
        self.assertIn(out["status"], ("no_match", "ambiguous", "possible"))

    def test_next_question_helps_separate_candidates(self):
        answers = {"part": ["leaf"], "leaf_top": ["yellow_spots"], "spot_look": ["unknown"]}
        suggestion = dg.suggest_next_question(self.kb, "grape", answers)
        self.assertIsNotNone(suggestion)
        self.assertNotIn(suggestion[0]["id"], dg.clean_answers(answers))
        self.assertEqual(suggestion[0]["id"], "leaf_under")

    def test_text_helpers(self):
        self.assertIn("الأوراق", dg.option_text(self.kb, "part", "leaf"))
        self.assertEqual(dg.option_text(self.kb, "part", "zzz"), "zzz")
        self.assertTrue(dg.question_text(self.kb, "leaf_top"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
