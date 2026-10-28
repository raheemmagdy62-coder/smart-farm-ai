"""Logic tests that run WITHOUT torch, Streamlit or internet.
They check that the code does what it says. They are NOT accuracy results:
any synthetic data used here says nothing about real plants."""
import io, json, sys, tempfile, unittest
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import crop_advisor as ca
import disease
import evaluate_plants as ev
import plant_model as pm
import weather as wx


class CatalogAndData(unittest.TestCase):
    def test_catalog(self):
        cat = pm.load_catalog()
        self.assertEqual(len(cat), 54)
        self.assertEqual(sorted(pm.core_keys(cat)), ["grape", "guava", "pear"])
        self.assertEqual(len({s.key for s in cat}), len(cat))
        cands = pm.build_candidates(cat)
        self.assertEqual(len(cands), sum(len(s.scientific) for s in cat) + len(pm.NON_PLANT_PROMPTS))

    def test_fao_tables_parse(self):
        rows = ca.load_fao56_rows()
        self.assertEqual(len(rows), 37)
        for r in rows:
            self.assertTrue(all(v > 0 for v in r["stages"]), r["crop_key"])
            self.assertTrue(all(1 <= m <= 12 for m in r["plant_months"]), r["crop_key"])
            self.assertTrue(all(0 < k < 2 for k in r["kc"]), r["crop_key"])
        sal = ca.load_salinity()
        self.assertEqual(sal["guava"]["threshold"], 4.7)
        self.assertEqual(sal["grape"]["rating"], "MS")
        self.assertIsNone(sal["pear"]["threshold"])
        self.assertEqual(len(ca.load_perennial_kc()), 18)
        self.assertIn("grape", ca.load_care_notes())


class Salinity(unittest.TestCase):
    def test_relative_yield(self):
        self.assertEqual(ca.relative_yield_percent(6.0, 6.0, 7.1), 100.0)
        self.assertAlmostEqual(ca.relative_yield_percent(10.0, 6.0, 7.1), 100 - 7.1 * 4)
        self.assertEqual(ca.relative_yield_percent(40.0, 6.0, 7.1), 0.0)
        self.assertAlmostEqual(ca.ece_from_ecw(2.0), 3.0)

    def test_qualitative_has_no_number(self):
        sal = ca.load_salinity()
        a = ca.salinity_assessment(sal["pear"], 5.0)
        self.assertIsNone(a["yr"]); self.assertTrue(a["estimate"])
        self.assertFalse(ca.salinity_assessment(None, 5.0)["available"])


class WaterAndSeason(unittest.TestCase):
    def test_kc_curve_and_water(self):
        stages, kc = [30, 40, 40, 25], [0.6, 1.15, 0.8]
        curve = ca.kc_curve(stages, kc)
        self.assertEqual(len(curve), 135)
        self.assertAlmostEqual(curve[29], 0.6); self.assertAlmostEqual(curve[69], 1.15)
        self.assertAlmostEqual(curve[-1], 0.8)
        normals = {m: {"et0_mean_daily": 5.0, "rain_total_mm": 0.0} for m in range(1, 13)}
        row = {"stages": stages, "kc": kc}
        out = ca.water_need(row, 1, normals)
        l_ini, l_dev, l_mid, l_late = stages
        expected = 5.0 * (l_ini * kc[0]
                          + l_dev * (kc[0] + (kc[1] - kc[0]) * (l_dev + 1) / (2 * l_dev))
                          + l_mid * kc[1]
                          + l_late * (kc[1] + (kc[2] - kc[1]) * (l_late + 1) / (2 * l_late)))
        self.assertAlmostEqual(out["etc_mm"], expected, places=6)
        self.assertEqual(out["days"], 135)

    def test_rain_reduces_net_and_missing_months(self):
        normals = {m: {"et0_mean_daily": 4.0, "rain_total_mm": 30.0} for m in range(1, 13)}
        out = ca.water_need({"stages": [10, 10, 10, 10], "kc": [1.0, 1.0, 1.0]}, 3, normals)
        self.assertGreater(out["rain_mm"], 0); self.assertLess(out["net_mm"], out["etc_mm"])
        self.assertIsNone(ca.water_need({"stages": [10, 10, 10, 10], "kc": [1.0, 1.0, 1.0]}, 3, {}))

    def test_month_distance(self):
        self.assertEqual(ca.month_distance(10, [10, 11]), 0)
        self.assertEqual(ca.month_distance(12, [1]), 1)
        self.assertEqual(ca.month_distance(10, [3, 4, 5]), 5)

    def test_advise_october(self):
        normals = {m: {"et0_mean_daily": 4.0, "rain_total_mm": 5.0} for m in range(1, 13)}
        res = ca.advise(normals, date(2026, 10, 27))
        names = {r["crop_key"] for r in res["candidates"]}
        self.assertIn("tomato", names); self.assertIn("lentil", names); self.assertIn("wheat", names)
        self.assertTrue(all(r["distance"] <= 1 for r in res["candidates"]))
        self.assertNotIn("cotton", names)
        cotton = [r for r in res["out_of_season"] if r["crop_key"] == "cotton"][0]
        self.assertEqual(cotton["distance"], 5)

    def test_advise_salinity_order_and_flags(self):
        normals = {m: {"et0_mean_daily": 4.0, "rain_total_mm": 5.0} for m in range(1, 13)}
        res = ca.advise(normals, date(2026, 10, 27), ece=12.0)
        d0 = [r for r in res["candidates"] if r["distance"] == 0]
        yrs = [r["salinity"]["yr"] for r in d0 if r["salinity"].get("yr") is not None]
        self.assertEqual(yrs, sorted(yrs, reverse=True))
        self.assertTrue(any("لا تعتمد" in f or "أعلى بكثير" in f for r in d0 for f in r["flags"]))

    def test_advise_without_climate_data(self):
        res = ca.advise({}, date(2026, 10, 27))
        self.assertTrue(all(r["water"] is None and r["flags"] for r in res["candidates"]))

    def test_missing_info(self):
        gaps = ca.missing_information(False, "غير معروف", "غير معروف", False)
        self.assertGreaterEqual(len(gaps), 6)


class PlantModelLogic(unittest.TestCase):
    def make_bundle(self, seed=0, dim=32):
        rng = np.random.default_rng(seed)
        cat = pm.load_catalog()
        cands = pm.build_candidates(cat)
        mat = rng.normal(size=(len(cands), dim)); mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        keys = [k for k, _ in cands]
        state = {"target": None}
        def encode_image(_img):
            v = state["target"]; return v / np.linalg.norm(v)
        def encode_text(prompts):
            v = rng.normal(size=(len(prompts), dim)); return v / np.linalg.norm(v, axis=1, keepdims=True)
        return {"text_matrix": mat.astype(np.float32), "keys": keys, "logit_scale": 100.0,
                "encode_image": encode_image, "encode_text": encode_text}, state, mat, keys

    def test_classify_picks_nearest(self):
        bundle, state, mat, keys = self.make_bundle()
        i = keys.index("guava"); state["target"] = mat[i] + 0.01
        res = pm.classify(object(), bundle)
        self.assertEqual(res["ranked"][0][0], "guava")
        self.assertAlmostEqual(sum(s for _, s in res["ranked"]), 1.0, places=6)
        self.assertGreater(res["max_similarity"], 0.9)
        self.assertEqual(res["embedding"].shape[0], 32)

    def test_extra_species(self):
        bundle, state, mat, keys = self.make_bundle()
        state["target"] = np.ones(32)
        res = pm.classify(object(), bundle, extra_species=["Ficus sycomorus", " "])
        self.assertIn(pm.CUSTOM_PREFIX + "Ficus sycomorus", [k for k, _ in res["ranked"]])
        self.assertEqual(len(res["ranked"]), len({*keys}) + 1)

    def test_decide_statuses(self):
        core = {"grape", "guava", "pear"}
        d = lambda r: pm.decide(r, 0.3, core)["status"]
        self.assertEqual(d([("grape", .90), ("pear", .05)]), "likely")
        self.assertEqual(d([("grape", .70), ("pear", .20)]), "uncertain")
        self.assertEqual(d([("grape", .85), ("pear", .80 - .0)]), "uncertain")  # small margin
        self.assertEqual(d([("apricot", .95), ("grape", .02)]), "other")
        self.assertEqual(d([("not_plant", .95), ("grape", .02)]), "not_plant")
        self.assertEqual(d([("grape", .30), ("pear", .25)]), "unknown")

    def test_display_names(self):
        idx = pm.species_index(pm.load_catalog())
        self.assertEqual(pm.display_names("apricot", idx), ("المشمش", "Apricot"))
        self.assertEqual(pm.display_names("custom:Foo bar", idx), ("Foo bar", "Foo bar"))
        self.assertEqual(pm.display_names("not_plant", idx)[1], "Not a plant")

    def test_quality_notes(self):
        from PIL import Image
        self.assertTrue(pm.image_quality_notes(Image.new("RGB", (50, 50), (128, 128, 128))))
        self.assertTrue(any("مظلمة" in n for n in pm.image_quality_notes(Image.new("RGB", (300, 300), (5, 5, 5)))))
        self.assertEqual(pm.image_quality_notes(Image.new("RGB", (300, 300), (120, 140, 90))), [])


class WeatherLogic(unittest.TestCase):
    def fake_opener(self, payload):
        def opener(url):
            self.last_url = url
            return io.BytesIO(json.dumps(payload).encode("utf-8"))
        return opener

    def test_normals(self):
        days, tmax, tmin, tmean, rain, et0 = [], [], [], [], [], []
        for y in (2023, 2024):
            for d in range(1, 32):
                days.append(f"{y}-01-{d:02d}"); tmax.append(20.0); tmin.append(-1.0 if d <= 5 else 5.0)
                tmean.append(12.0); rain.append(1.0); et0.append(2.0)
        n = wx.monthly_normals({"time": days, "temperature_2m_max": tmax, "temperature_2m_min": tmin,
                                "temperature_2m_mean": tmean, "precipitation_sum": rain,
                                "et0_fao_evapotranspiration": et0})
        self.assertEqual(list(n), [1])
        self.assertAlmostEqual(n[1]["rain_total_mm"], 31.0)
        self.assertAlmostEqual(n[1]["frost_days_per_year"], 5.0)
        self.assertAlmostEqual(n[1]["et0_mean_daily"], 2.0); self.assertEqual(n[1]["abs_min"], -1.0)

    def test_normals_with_missing_values(self):
        n = wx.monthly_normals({"time": ["2024-03-01", "2024-03-02"], "temperature_2m_max": [None, 30.0],
                                "temperature_2m_min": [None, 15.0], "temperature_2m_mean": [None, 22.0],
                                "precipitation_sum": [None, 0.0], "et0_fao_evapotranspiration": [None, 5.0]})
        self.assertAlmostEqual(n[3]["tmax_mean"], 30.0)

    def test_geocode_filters_country(self):
        payload = {"results": [
            {"name": "المنصورة", "admin1": "الدقهلية", "country": "مصر", "country_code": "EG", "latitude": 31.04, "longitude": 31.38},
            {"name": "Mansoura", "admin1": "X", "country": "Other", "country_code": "ZZ", "latitude": 1.0, "longitude": 2.0}]}
        res = wx.geocode("المنصورة", opener=self.fake_opener(payload))
        self.assertEqual([p["country_code"] for p in res], ["EG"])
        self.assertIn("geocoding-api.open-meteo.com", self.last_url)
        self.assertEqual(wx.geocode("x", opener=self.fake_opener({})), [])

    def test_forecast_and_alerts(self):
        payload = {"current": {"temperature_2m": 22.0, "relative_humidity_2m": 85},
                   "daily": {"time": ["2026-10-28", "2026-10-29"], "temperature_2m_max": [39.0, 30.0],
                             "temperature_2m_min": [1.0, 10.0], "precipitation_sum": [12.0, 0.0],
                             "et0_fao_evapotranspiration": [3.0, 3.0]}}
        fc = wx.fetch_forecast(30.0, 31.0, opener=self.fake_opener(payload))
        self.assertIn("fetched_at", fc)
        self.assertEqual(len(wx.general_alerts(fc)), 4)
        calm = {"current": {"temperature_2m": 25.0, "relative_humidity_2m": 40},
                "daily": {"temperature_2m_max": [30.0], "temperature_2m_min": [15.0], "precipitation_sum": [0.0]}}
        self.assertEqual(wx.general_alerts(calm), [])

    def test_bad_response_raises(self):
        with self.assertRaises(ValueError):
            wx.fetch_forecast(1, 2, opener=self.fake_opener({"nothing": 1}))


class DiseaseLogic(unittest.TestCase):
    def synth(self, n_per=40, dim=24, seed=1):
        rng = np.random.default_rng(seed)
        centers = rng.normal(size=(3, dim)) * 2
        x = np.vstack([centers[c] + rng.normal(size=(n_per, dim)) for c in range(3)])
        y = np.repeat(np.arange(3), n_per)
        return x, y

    def test_split_has_no_group_leak(self):
        labels = np.repeat([0, 1], 30)
        groups = [f"{l}:{i // 3}" for i, l in enumerate(labels)]
        tr, va, te = disease.split_indices(labels, groups, seed=3)
        g = lambda idx: {groups[i] for i in idx}
        self.assertFalse(g(tr) & g(va)); self.assertFalse(g(tr) & g(te)); self.assertFalse(g(va) & g(te))
        self.assertEqual(len(tr) + len(va) + len(te), 60)

    def test_pipeline_on_synthetic_clusters(self):
        x, y = self.synth()
        res = disease.fit_and_evaluate(x, y, ["healthy", "black_rot", "mildew"], seed=0)
        self.assertGreater(res["test_report"]["accuracy"], 0.9)   # synthetic: pipeline check only
        self.assertEqual(sum(res["sizes"].values()), 120)
        self.assertEqual(len(res["abstention"]), 5)

    def test_report_metrics_exact(self):
        rep = disease.classification_report(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), ["a", "b"])
        self.assertAlmostEqual(rep["accuracy"], 0.75)
        self.assertAlmostEqual(rep["per_class"]["a"]["recall"], 0.5)
        self.assertAlmostEqual(rep["per_class"]["b"]["precision"], 2 / 3)
        self.assertEqual(rep["confusion"], [[1, 1], [0, 2]])

    def test_too_few_photos_raises(self):
        with self.assertRaises(ValueError):
            disease.fit_and_evaluate(np.zeros((4, 3)), np.array([0, 0, 1, 1]), ["a", "b"])

    def test_save_load_interpret(self):
        x, y = self.synth()
        names = ["healthy", "black_rot", "mildew"]
        res = disease.fit_and_evaluate(x, y, names, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            disease.save_head("demo", res, names, {"model_id": "test"}, models_dir=tmp)
            head = disease.DiseaseHead.load("demo", models_dir=tmp)
            self.assertIsNone(disease.DiseaseHead.load("missing", models_dir=tmp))
            self.assertEqual(head.classes, names)
            self.assertIn("test", head.metrics)
            out = head.interpret(x[0])
            self.assertEqual(out["top"], "healthy"); self.assertEqual(out["status"], "healthy")
            far = head.interpret(np.zeros(x.shape[1]), min_probability=0.999999)
            self.assertEqual(far["status"], "uncertain")
            sick = head.interpret(x[-1])
            self.assertEqual(sick["status"], "possible_symptoms")


class EvaluateSummary(unittest.TestCase):
    def test_summarize(self):
        mk = lambda t, p, s, score: {"true": t, "pred": p, "status": s, "score": score}
        recs = [mk("grape", "grape", "likely", .95), mk("grape", "guava", "uncertain", .6),
                mk("other", "grape", "likely", .9), mk("other", "other", "other", .8),
                mk("pear", "pear", "likely", .85)]
        s = ev.summarize(recs)
        self.assertEqual(s["n"], 5); self.assertAlmostEqual(s["accuracy_top1"], 3 / 5)
        self.assertAlmostEqual(s["false_accept_rate_on_other"], 0.5)
        self.assertAlmostEqual(s["coverage_likely"], 3 / 5); self.assertAlmostEqual(s["accuracy_when_likely"], 2 / 3)
        self.assertAlmostEqual(s["per_class"]["grape"]["precision"], 0.5)
        self.assertAlmostEqual(s["per_class"]["grape"]["recall"], 0.5)
        self.assertEqual(sum(c["n"] for c in s["calibration"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
