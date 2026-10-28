"""Smoke test: runs app.py top to bottom with a FAKE streamlit module.
It catches NameError / TypeError / wrong function calls in the script.
It does NOT test how the page looks (that needs a real browser)."""
import io, runpy, sys, types, unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_png(color=(70, 140, 60), size=(320, 240)):
    buf = io.BytesIO(); Image.new("RGB", size, color).save(buf, "PNG"); return buf.getvalue()


class FakeUpload:
    def __init__(self, data): self._d = data
    def getvalue(self): return self._d


def fake_streamlit(upload=None, button=False, query="المنصورة", widgets=None):
    widgets = widgets or {}
    st = types.ModuleType("streamlit")
    calls = {"error": [], "success": [], "warning": [], "info": [], "markdown": [], "code": []}
    def rec(name):
        def f(*a, **k): calls[name].append(a[0] if a else ""); return MagicMock()
        return f
    for n in ("error", "success", "warning", "info", "markdown", "code"):
        setattr(st, n, rec(n))
    def passthrough_decorator(*a, **k):
        if a and callable(a[0]): return a[0]
        return lambda fn: fn
    st.cache_resource = passthrough_decorator
    st.cache_data = passthrough_decorator
    st.session_state = {}
    st.set_page_config = MagicMock()
    st.columns = lambda n, **k: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
    st.tabs = lambda labels: [MagicMock() for _ in labels]
    for n in ("container", "expander", "spinner"):
        setattr(st, n, lambda *a, **k: MagicMock())
    st.file_uploader = lambda *a, **k: upload
    st.camera_input = lambda *a, **k: None
    st.checkbox = lambda *a, **k: False
    st.button = lambda *a, **k: button
    st.selectbox = lambda label, options, **k: widgets.get(k.get("key"), options[0])
    st.radio = lambda label, options, **k: options[0]
    st.text_input = lambda label, **k: query if "منطقة" in label else k.get("value", "")
    st.number_input = lambda *a, **k: k.get("value", 0.0)
    st.multiselect = lambda label, options, **k: widgets.get(k.get("key"), [])
    for n in ("subheader", "caption", "write", "image", "progress", "metric", "header", "title"):
        setattr(st, n, MagicMock())
    return st, calls


def run_app(st):
    sys.modules["streamlit"] = st
    for m in ("app",):
        sys.modules.pop(m, None)
    return runpy.run_path(str(ROOT / "app.py"), run_name="__main__")


class AppSmoke(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("streamlit", None)

    def test_empty_page(self):
        st, calls = fake_streamlit()
        run_app(st)
        self.assertEqual(calls["error"], [])

    def test_invalid_file_shows_friendly_error(self):
        st, calls = fake_streamlit(upload=FakeUpload(b"not an image"))
        run_app(st)
        self.assertTrue(any("تعذّرت قراءة الصورة" in e for e in calls["error"]))

    def test_empty_and_oversized_file(self):
        st, calls = fake_streamlit(upload=FakeUpload(b""))
        run_app(st); self.assertTrue(any("فارغ" in e for e in calls["error"]))
        st, calls = fake_streamlit(upload=FakeUpload(b"x" * (11 * 1024 * 1024)))
        run_app(st); self.assertTrue(any("كبير جدًا" in e for e in calls["error"]))

    def test_gif_renamed_is_rejected(self):
        buf = io.BytesIO(); Image.new("RGB", (50, 50)).save(buf, "GIF")
        st, calls = fake_streamlit(upload=FakeUpload(buf.getvalue()))
        run_app(st); self.assertTrue(any("غير مدعوم" in e for e in calls["error"]))

    def test_missing_ai_libraries_gives_install_message(self):
        import plant_model as pm
        def boom(*a, **k): raise ImportError("No module named 'open_clip'")
        original = pm.load_model; pm.load_model = boom
        try:
            st, calls = fake_streamlit(upload=FakeUpload(make_png()))
            run_app(st)
        finally:
            pm.load_model = original
        self.assertTrue(any("غير مثبّتة" in e for e in calls["error"]))
        self.assertTrue(any("pip install" in c for c in calls["code"]))

    def test_model_failure_shows_generic_error(self):
        import plant_model as pm
        def boom(*a, **k): raise RuntimeError("download failed")
        original = pm.load_model; pm.load_model = boom
        try:
            st, calls = fake_streamlit(upload=FakeUpload(make_png()))
            run_app(st)
        finally:
            pm.load_model = original
        self.assertTrue(any("تعذّر تحميل النموذج" in e for e in calls["error"]))

    def test_full_result_path_with_fake_model(self):
        import plant_model as pm
        cat = pm.load_catalog(); cands = pm.build_candidates(cat)
        rng = np.random.default_rng(0); dim = 16
        mat = rng.normal(size=(len(cands), dim)); mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        keys = [k for k, _ in cands]; target = mat[keys.index("grape")]
        bundle = {"text_matrix": mat.astype(np.float32), "keys": keys, "logit_scale": 100.0, "catalog": cat,
                  "encode_image": lambda img: target, "encode_text": lambda p: np.tile(target, (len(p), 1))}
        original = pm.load_model; pm.load_model = lambda *a, **k: bundle
        try:
            st, calls = fake_streamlit(upload=FakeUpload(make_png()))
            run_app(st)
        finally:
            pm.load_model = original
        self.assertEqual(calls["error"], [])
        self.assertTrue(any("النتيجة المحتملة" in s and "العنب" in s for s in calls["success"]))
        self.assertTrue(any("لا يوجد نموذج أمراض" in i for i in calls["info"]))   # no head installed

    def test_symptom_tab_insufficient_by_default(self):
        st, calls = fake_streamlit()
        run_app(st)
        self.assertTrue(any("غير كافية" in w for w in calls["warning"]))
        self.assertEqual(calls["error"], [])

    def test_symptom_tab_with_answers_shows_possible_condition(self):
        widgets = {"sym_grape_part": ["leaf"], "sym_grape_leaf_top": "yellow_spots", "sym_grape_leaf_under": "white_cottony"}
        st, calls = fake_streamlit(widgets=widgets)
        run_app(st)
        self.assertTrue(any("احتمال متقدم" in s for s in calls["success"]))
        self.assertTrue(any("البياض الزغبي" in str(m) for m in calls["markdown"]))
        self.assertEqual(calls["error"], [])

    def test_symptom_tab_uses_identified_plant(self):
        import plant_model as pm
        cat = pm.load_catalog(); cands = pm.build_candidates(cat)
        rng = np.random.default_rng(1); dim = 16
        mat = rng.normal(size=(len(cands), dim)); mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        keys = [k for k, _ in cands]; target = mat[keys.index("pear")]
        bundle = {"text_matrix": mat.astype(np.float32), "keys": keys, "logit_scale": 100.0, "catalog": cat,
                  "encode_image": lambda img: target, "encode_text": lambda p: np.tile(target, (len(p), 1))}
        original = pm.load_model; pm.load_model = lambda *a, **k: bundle
        try:
            st, calls = fake_streamlit(upload=FakeUpload(make_png()))
            run_app(st)
        finally:
            pm.load_model = original
        self.assertEqual(st.session_state.get("identified_core"), "pear")

    def test_advisor_button_with_fake_weather(self):
        import weather as wx
        days = [f"{y}-{m:02d}-{d:02d}" for y in (2024, 2025) for m in range(1, 13) for d in range(1, 29)]
        n = len(days)
        hist = {"daily": {"time": days, "temperature_2m_max": [30.0] * n, "temperature_2m_min": [15.0] * n,
                          "temperature_2m_mean": [22.0] * n, "precipitation_sum": [0.1] * n,
                          "et0_fao_evapotranspiration": [4.0] * n},
                "fetched_at": "t", "period": "p"}
        fc = {"current": {"temperature_2m": 25.0, "relative_humidity_2m": 50},
              "daily": {"time": ["d"], "temperature_2m_max": [30.0], "temperature_2m_min": [15.0], "precipitation_sum": [0.0]},
              "fetched_at": "t"}
        place = {"name": "المنصورة", "admin1": "الدقهلية", "country": "مصر", "country_code": "EG",
                 "latitude": 31.04, "longitude": 31.38, "elevation": 8}
        saved = (wx.geocode, wx.fetch_forecast, wx.fetch_climate_history)
        wx.geocode = lambda q, **k: [place]; wx.fetch_forecast = lambda *a, **k: fc; wx.fetch_climate_history = lambda *a, **k: hist
        try:
            st, calls = fake_streamlit(button=True)
            run_app(st)
        finally:
            wx.geocode, wx.fetch_forecast, wx.fetch_climate_history = saved
        self.assertEqual(calls["error"], [])
        self.assertTrue(any("المنصورة" in m for m in [str(x) for x in calls["markdown"]]))
        self.assertTrue(st.session_state["advisor"]["advice"]["candidates"])

    def test_advisor_network_failure_is_reported(self):
        import urllib.error, weather as wx
        saved = wx.geocode
        def fail(*a, **k): raise urllib.error.URLError("no network")
        wx.geocode = fail
        try:
            st, calls = fake_streamlit(button=True)
            run_app(st)
        finally:
            wx.geocode = saved
        self.assertTrue(any("تعذّر الوصول إلى خدمة الطقس" in e for e in calls["error"]))
        self.assertNotIn("advisor", st.session_state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
