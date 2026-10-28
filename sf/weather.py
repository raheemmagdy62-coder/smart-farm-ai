"""weather.py - free weather and climate data from Open-Meteo (no API key).

Only the Python standard library is used. Attribution is REQUIRED by Open-Meteo:
show a link such as "Weather data by Open-Meteo" next to any weather result.
Free use is for NON-COMMERCIAL projects. Re-check their terms if this becomes a product.

Every function returns real data from the service or raises an error.
It never invents numbers and never returns old data as if it were fresh.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT_SECONDS = 20
ATTRIBUTION_URL = "https://open-meteo.com/"

# Rough alert limits for GENERAL awareness only. Not crop-specific, not measured.
# They should be reviewed by an agronomist before real use.
FROST_MIN_C = 2.0
HEAT_MAX_C = 38.0
HEAVY_RAIN_MM = 10.0
HUMID_PERCENT = 80
FUNGAL_TEMP_RANGE = (15.0, 30.0)

NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError)


def _get_json(base_url, params, opener=None):
    """GET a URL and parse JSON. `opener` can be replaced in tests."""
    url = base_url + "?" + urllib.parse.urlencode(params)
    open_url = opener or (lambda u: urllib.request.urlopen(u, timeout=TIMEOUT_SECONDS))
    with open_url(url) as response:
        return json.load(response)


def _now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def geocode(name, country_code="EG", count=10, opener=None):
    """Find places by name. Returns a list of dicts (may be empty).

    When country_code is given, results from that country are kept; if there
    are none, the unfiltered results are returned so the user can still choose.
    """
    data = _get_json(
        GEOCODING_URL,
        {"name": name, "count": count, "language": "ar", "format": "json"},
        opener,
    )
    places = []
    for item in data.get("results") or []:
        places.append(
            {
                "name": item.get("name", name),
                "admin1": item.get("admin1", ""),
                "country": item.get("country", ""),
                "country_code": item.get("country_code", ""),
                "latitude": float(item["latitude"]),
                "longitude": float(item["longitude"]),
                "elevation": item.get("elevation"),
            }
        )
    if country_code:
        filtered = [p for p in places if p["country_code"] == country_code]
        if filtered:
            return filtered
    return places


def fetch_forecast(latitude, longitude, days=16, opener=None):
    """Current weather + daily forecast. Returns the raw API data plus fetched_at."""
    data = _get_json(
        FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration",
            "timezone": "auto",
            "forecast_days": days,
        },
        opener,
    )
    if "daily" not in data or "time" not in data["daily"]:
        raise ValueError("Unexpected forecast response")
    data["fetched_at"] = _now_utc()
    return data


def fetch_climate_history(latitude, longitude, years=6, today=None, opener=None):
    """Daily history (ERA5 reanalysis via Open-Meteo) for the last full years."""
    today = today or date.today()
    start = f"{today.year - years}-01-01"
    end = f"{today.year - 1}-12-31"
    data = _get_json(
        ARCHIVE_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start,
            "end_date": end,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                     "precipitation_sum,et0_fao_evapotranspiration",
            "timezone": "auto",
        },
        opener,
    )
    if "daily" not in data or "time" not in data["daily"]:
        raise ValueError("Unexpected archive response")
    data["fetched_at"] = _now_utc()
    data["period"] = f"{start} -> {end}"
    return data


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def monthly_normals(daily):
    """Per-month climate normals from a daily series (pure function).

    daily: dict with 'time' (YYYY-MM-DD) and these lists (None values allowed):
           temperature_2m_max, temperature_2m_min, temperature_2m_mean,
           precipitation_sum, et0_fao_evapotranspiration
    Returns {month: {...}} for months 1..12 that have data.
    """
    times = daily["time"]
    buckets = {m: {"tmax": [], "tmin": [], "tmean": [], "et0": [], "rain_by_year": {}, "years": set()}
               for m in range(1, 13)}
    for i, day in enumerate(times):
        year, month = int(day[:4]), int(day[5:7])
        b = buckets[month]
        b["years"].add(year)
        b["tmax"].append(daily["temperature_2m_max"][i])
        b["tmin"].append(daily["temperature_2m_min"][i])
        b["tmean"].append(daily["temperature_2m_mean"][i])
        b["et0"].append(daily["et0_fao_evapotranspiration"][i])
        rain = daily["precipitation_sum"][i]
        if rain is not None:
            b["rain_by_year"][year] = b["rain_by_year"].get(year, 0.0) + rain

    normals = {}
    for month, b in buckets.items():
        if not b["years"]:
            continue
        tmin_clean = [v for v in b["tmin"] if v is not None]
        rain_totals = list(b["rain_by_year"].values())
        normals[month] = {
            "tmax_mean": _mean(b["tmax"]),
            "tmin_mean": _mean(b["tmin"]),
            "tmean": _mean(b["tmean"]),
            "et0_mean_daily": _mean(b["et0"]),
            "rain_total_mm": _mean(rain_totals) if rain_totals else 0.0,
            "abs_min": min(tmin_clean) if tmin_clean else None,
            "frost_days_per_year": (sum(1 for v in tmin_clean if v <= 0.0) / len(b["years"])) if tmin_clean else None,
            "years": len(b["years"]),
        }
    return normals


def general_alerts(forecast):
    """General (not crop-specific) alerts from the forecast. Returns Arabic strings."""
    daily = forecast["daily"]
    current = forecast.get("current") or {}
    mins = [v for v in daily.get("temperature_2m_min", []) if v is not None]
    maxs = [v for v in daily.get("temperature_2m_max", []) if v is not None]
    rains = [v for v in daily.get("precipitation_sum", []) if v is not None]
    alerts = []

    if mins and min(mins) <= FROST_MIN_C:
        alerts.append(f"توقعات بحرارة صغرى منخفضة حتى {min(mins):.1f}°م خلال الأيام القادمة؛ قد تتأثر النباتات الحساسة للبرد.")
    if maxs and max(maxs) >= HEAT_MAX_C:
        alerts.append(f"توقعات بحرارة عظمى مرتفعة حتى {max(maxs):.1f}°م؛ راقب إجهاد الحرارة واحتياج النبات للماء.")
    if rains and max(rains) >= HEAVY_RAIN_MM:
        alerts.append(f"توقعات بأمطار تصل إلى {max(rains):.0f} مم في يوم واحد؛ قد ترتفع مخاطر الأمراض الفطرية مع الرطوبة.")
    humidity = current.get("relative_humidity_2m")
    temp = current.get("temperature_2m")
    if (
        humidity is not None
        and temp is not None
        and humidity >= HUMID_PERCENT
        and FUNGAL_TEMP_RANGE[0] <= temp <= FUNGAL_TEMP_RANGE[1]
    ):
        alerts.append("الرطوبة الحالية مرتفعة مع حرارة معتدلة، وهي ظروف قد تناسب انتشار بعض الأمراض الفطرية. هذا تنبيه عام وليس تشخيصًا.")
    return alerts
