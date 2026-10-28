"""crop_advisor.py - Smart Crop Advisor (version 0.1).

What this engine does, and what it deliberately does NOT do:

* Season check: compares the current month with the planting months listed in
  FAO Irrigation and Drainage Paper 56, Table 11 (data/fao56_crops.csv).
  Those rows come from climates that are SIMILAR to Egyptian ones (arid,
  Mediterranean). Only cotton and green beans are labeled "Egypt" in the table.
  FAO itself says the values are indicative and local data should be used.
* Water need: ETc = Kc x ET0 summed over the crop cycle (FAO-56 method).
  ET0 comes from real local climate data (weather.py). Kc is the STANDARD value;
  the humidity/wind adjustment (FAO-56 eq. 62) is NOT applied.
* Salinity: relative yield from the FAO threshold-slope model
  Yr = 100 - b (ECe - a)   (FAO Irrigation and Drainage Paper 61, Annex 1,
  data from Maas & Grattan 1999). Crops with only a qualitative rating are
  shown with the rating, never with an invented number.
* NOT used: soil texture, drainage, pH, groundwater, costs, yields, prices.
  No sourced rule set for these is included yet, so the app lists them as
  MISSING INFORMATION instead of pretending.

The output is a transparent list with reasons. It is not a yield forecast and
not a final recommendation.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# FAO Paper 29: ECe of the root zone is roughly 1.5 x the irrigation water EC
# (ECw) at a typical leaching fraction. APPROXIMATE: use a measured ECe if possible.
ECE_FROM_ECW_FACTOR = 1.5
CLIMATE_PRIORITY = {"egypt": 0, "arid": 1, "mediterranean": 2}

MONTH_NAMES_AR = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
                  "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]
RATING_AR = {"S": "حساس", "MS": "حساس نسبيًا", "MT": "متوسط التحمل", "T": "متحمل"}


def _float(text):
    text = (text or "").strip()
    return float(text) if text else None


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
def load_salinity(path=DATA_DIR / "crop_salinity_fao.csv"):
    table = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            table[row["key"]] = {
                "threshold": _float(row["ece_threshold_dsm"]),
                "slope": _float(row["slope_pct_per_dsm"]),
                "rating": row["rating"].strip(),
                "estimate": row["rating_is_estimate"].strip() == "1",
                "based_on": row["based_on"].strip(),
                "note": row["note"].strip(),
            }
    return table


def load_fao56_rows(path=DATA_DIR / "fao56_crops.csv"):
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if None in row:
                raise ValueError(f"Too many columns in fao56 row: {row.get('crop_key')}")
            rows.append(
                {
                    "crop_key": row["crop_key"].strip(),
                    "name_ar": row["name_ar"].strip(),
                    "name_en": row["name_en"].strip(),
                    "region_label": row["region_label"].strip(),
                    "climate_class": row["climate_class"].strip(),
                    "plant_months": [int(m) for m in row["plant_months"].split(";") if m.strip()],
                    "stages": [int(row[k]) for k in ("l_ini", "l_dev", "l_mid", "l_late")],
                    "kc": [float(row[k]) for k in ("kc_ini", "kc_mid", "kc_end")],
                    "source_ref": row["source_ref"].strip(),
                    "notes": row["notes"].strip(),
                }
            )
    return rows


def load_perennial_kc(path=DATA_DIR / "fao56_perennial_kc.csv"):
    table = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            table[row["key"]] = {
                "name_en": row["name_en"],
                "kc": [float(row["kc_ini"]), float(row["kc_mid"]), float(row["kc_end"])],
                "source_ref": row["source_ref"],
                "notes": row["notes"],
            }
    return table


def load_care_notes(path=DATA_DIR / "care_notes.csv"):
    notes = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            notes.setdefault(row["key"], []).append(row)
    return notes


# ---------------------------------------------------------------
# Salinity
# ---------------------------------------------------------------
def ece_from_ecw(ecw):
    """Approximate root-zone ECe (dS/m) from irrigation water EC (dS/m)."""
    return ecw * ECE_FROM_ECW_FACTOR


def relative_yield_percent(ece, threshold, slope):
    """FAO threshold-slope model: Yr = 100 - b (ECe - a), limited to 0..100."""
    if ece <= threshold:
        return 100.0
    return max(0.0, 100.0 - slope * (ece - threshold))


def salinity_assessment(entry, ece):
    """Describe how a crop's salinity data relates to a given ECe (or None)."""
    if entry is None:
        return {"available": False}
    result = {
        "available": True,
        "rating": entry["rating"],
        "rating_ar": RATING_AR.get(entry["rating"], entry["rating"]),
        "estimate": entry["estimate"],
        "threshold": entry["threshold"],
        "slope": entry["slope"],
        "yr": None,
        "note": entry["note"],
    }
    if ece is not None and entry["threshold"] is not None and entry["slope"] is not None:
        result["yr"] = relative_yield_percent(ece, entry["threshold"], entry["slope"])
    return result


# ---------------------------------------------------------------
# Water need (FAO-56 single crop coefficient method)
# ---------------------------------------------------------------
def kc_curve(stages, kc):
    """Daily Kc for the whole cycle: flat ini, linear dev, flat mid, linear late."""
    l_ini, l_dev, l_mid, l_late = stages
    kc_ini, kc_mid, kc_end = kc
    curve = [kc_ini] * l_ini
    curve += [kc_ini + (kc_mid - kc_ini) * (i + 1) / l_dev for i in range(l_dev)]
    curve += [kc_mid] * l_mid
    curve += [kc_mid + (kc_end - kc_mid) * (i + 1) / l_late for i in range(l_late)]
    return curve


def water_need(row, start_month, normals, start_day=15):
    """Estimate ETc, climatic rainfall and the difference over one crop cycle.

    Uses monthly normals (ET0 mean per day, rain per month). Returns None when
    the climate data does not cover every month of the cycle.
    """
    curve = kc_curve(row["stages"], row["kc"])
    month, day = start_month, start_day
    etc_total = rain_total = 0.0
    for kc in curve:
        info = normals.get(month)
        if info is None or info.get("et0_mean_daily") is None:
            return None
        etc_total += kc * info["et0_mean_daily"]
        rain_total += (info.get("rain_total_mm") or 0.0) / DAYS_IN_MONTH[month - 1]
        day += 1
        if day > DAYS_IN_MONTH[month - 1]:
            day = 1
            month = month % 12 + 1
    return {
        "days": len(curve),
        "etc_mm": etc_total,
        "rain_mm": rain_total,
        "net_mm": max(0.0, etc_total - rain_total),
    }


# ---------------------------------------------------------------
# Season matching and ranking
# ---------------------------------------------------------------
def month_distance(current_month, plant_months):
    """Smallest number of months from now until a planting month (0 = this month)."""
    return min((m - current_month) % 12 for m in plant_months)


def advise(normals, today=None, ece=None, max_distance=1, fao_rows=None, salinity=None):
    """Rank crops whose FAO-56 planting months are now or soon.

    Returns {"candidates": [...], "out_of_season": [...]}.
    Order: nearest planting month first; then, if ECe is known, higher
    estimated relative yield first; then by name. No hidden score is used.
    """
    today = today or date.today()
    fao_rows = fao_rows if fao_rows is not None else load_fao56_rows()
    salinity = salinity if salinity is not None else load_salinity()

    by_crop = {}
    for row in fao_rows:
        by_crop.setdefault(row["crop_key"], []).append(row)

    candidates, out_of_season = [], []
    for crop_key, rows in by_crop.items():
        scored = []
        for row in rows:
            dist = month_distance(today.month, row["plant_months"])
            scored.append((dist, CLIMATE_PRIORITY.get(row["climate_class"], 9), row))
        scored.sort(key=lambda item: (item[0], item[1]))
        dist, _, row = scored[0]
        start_month = (today.month - 1 + dist) % 12 + 1
        if dist not in (0, 1) and start_month not in row["plant_months"]:
            start_month = min(row["plant_months"], key=lambda m: (m - today.month) % 12)

        sal = salinity_assessment(salinity.get(crop_key), ece)
        flags = []
        water = water_need(row, start_month, normals) if normals else None
        if water is None:
            flags.append("تعذّر حساب احتياج المياه لأن بيانات المناخ ناقصة.")
        if not sal["available"]:
            flags.append("لا توجد بيانات ملوحة لهذا المحصول في الجدول المرجعي.")
        elif sal["yr"] is None and ece is not None:
            flags.append("بيانات الملوحة لهذا المحصول وصفية فقط (بدون قيمة رقمية).")
        if sal.get("available") and sal["yr"] is not None and sal["yr"] <= 0:
            flags.append("الملوحة المدخلة أعلى بكثير من حد تحمل هذا المحصول حسب نموذج FAO.")

        record = {
            "crop_key": crop_key,
            "name_ar": row["name_ar"],
            "name_en": row["name_en"],
            "distance": dist,
            "start_month": start_month,
            "region_label": row["region_label"],
            "climate_class": row["climate_class"],
            "cycle_days": sum(row["stages"]),
            "water": water,
            "salinity": sal,
            "flags": flags,
            "source_ref": row["source_ref"],
            "notes": row["notes"],
        }
        (candidates if dist <= max_distance else out_of_season).append(record)

    def sort_key(rec):
        yr = rec["salinity"].get("yr") if rec["salinity"].get("available") else None
        return (rec["distance"], 0 if yr is not None else 1, -(yr or 0.0), rec["name_ar"])

    candidates.sort(key=sort_key)
    out_of_season.sort(key=lambda rec: (rec["distance"], rec["name_ar"]))
    return {"candidates": candidates, "out_of_season": out_of_season}


def missing_information(ece_known, soil_texture, drainage, ph_known):
    """What the user did not provide, so the app can say what limits the advice."""
    gaps = []
    if not ece_known:
        gaps.append("ملوحة التربة (ECe) أو ملوحة ماء الري (EC) غير معروفة: لا يمكن تقدير تأثير الملوحة على المحصول.")
    if soil_texture in (None, "", "غير معروف"):
        gaps.append("قوام التربة غير معروف (رملية/طينية/...): لم تُستخدم في الترتيب لعدم وجود قواعد موثقة مضمنة بعد.")
    if drainage in (None, "", "غير معروف"):
        gaps.append("حالة الصرف غير معروفة: لم تُستخدم في الترتيب.")
    if not ph_known:
        gaps.append("درجة حموضة التربة (pH) غير معروفة: لم تُستخدم في الترتيب.")
    gaps.append("مصدر ومقدار مياه الري المتاح فعليًا غير مدخل: الأرقام تعرض احتياج المحصول وليس المتاح.")
    gaps.append("الصنف المحلي وتاريخ الحقل (المحصول السابق والآفات) غير معروفين.")
    gaps.append("لا تشمل النسخة الحالية التكلفة أو الإنتاجية أو الأسعار أو العائد المتوقع.")
    return gaps


def species_facts(key, salinity=None, perennial_kc=None, care_notes=None):
    """Sourced facts about one identified species (used by the plant tab)."""
    salinity = salinity if salinity is not None else load_salinity()
    perennial_kc = perennial_kc if perennial_kc is not None else load_perennial_kc()
    care_notes = care_notes if care_notes is not None else load_care_notes()
    return {
        "salinity": salinity_assessment(salinity.get(key), None),
        "kc": perennial_kc.get(key),
        "care_notes": care_notes.get(key, []),
    }
