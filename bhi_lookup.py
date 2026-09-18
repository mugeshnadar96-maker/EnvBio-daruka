"""
bhi_lookup.py
-------------
Implements the "bonus" geo-coordinate input path from the spec, for real:
given a (lat, lon) inside India, look up the actual CSIRO Biodiversity
Habitat Index (BHI v4, BILBI model) value and its 2000-2024 trend from the
GeoTIFF (25 annual bands, 2000-2024, ~1km resolution).

Locating the raster file, in priority order:
  1. The `raster_path` argument, if the caller passes one explicitly.
  2. The BHI_RASTER_PATH environment variable, if set.
  3. ./CSIRO_BHI_v4_India_All_Years.tif, next to this script.
  4. ./data/CSIRO_BHI_v4_India_All_Years.tif
  5. /mnt/user-data/uploads/CSIRO_BHI_v4_India_All_Years.tif (only present
     inside the Claude sandbox this project was originally built in -- NOT
     present on your machine unless you're running inside that same
     sandbox). If you're running this locally, put your own copy of the
     ~300MB raster at (3) or (4) instead, or pass raster_path= explicitly.
     The file is not bundled in the project zip because of its size.

IMPORTANT: earlier versions of this module returned None on every failure
mode (missing file, missing rasterio, point outside India, nodata pixel)
with no way to tell them apart, so geo-coordinate queries failed silently.
This version returns a dict with an "error" key explaining exactly what
went wrong, and conversation_agent.py surfaces that message to the user
instead of just silently skipping the lookup.
"""

import os
from typing import Dict

YEARS = list(range(2000, 2025))  # bands 1..25

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_PATHS = [
    os.environ.get("BHI_RASTER_PATH"),
    os.path.join(_SCRIPT_DIR, "CSIRO_BHI_v4_India_All_Years.tif"),
    os.path.join(_SCRIPT_DIR, "data", "CSIRO_BHI_v4_India_All_Years.tif"),
    "/mnt/user-data/uploads/CSIRO_BHI_v4_India_All_Years.tif",
]


def resolve_raster_path() -> str:
    """Returns the first candidate path that actually exists on disk, or
    the first non-None candidate (for a clear error message) if none do."""
    candidates = [p for p in _CANDIDATE_PATHS if p]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0] if candidates else "CSIRO_BHI_v4_India_All_Years.tif"


def _in_india_bbox(lat: float, lon: float) -> bool:
    # Matches the raster's own bounds: lon 68.18-97.18, lat 6.75-33.17
    return 6.75 <= lat <= 33.18 and 68.17 <= lon <= 97.18


def lookup_bhi(lat: float, lon: float, raster_path: str = None) -> Dict:
    """
    Returns one of:
      {"ok": True, "bhi_latest": float, "bhi_2000": float,
       "trend_2000_2024": float, "year_latest": int}
      {"ok": False, "error": "<human-readable reason>"}

    Always returns a dict -- callers check the "ok" key rather than testing
    for None, so a failed lookup is never silently indistinguishable from
    "no data at this point".
    """
    if raster_path is None:
        raster_path = resolve_raster_path()

    if not _in_india_bbox(lat, lon):
        return {"ok": False, "error": f"({lat}, {lon}) is outside the raster's India coverage "
                                       f"(lat 6.75-33.18, lon 68.17-97.18)."}

    if not os.path.exists(raster_path):
        return {"ok": False, "error": (
            f"Raster file not found at '{raster_path}'. This ~300MB file isn't bundled in the "
            f"project download -- copy your own copy of CSIRO_BHI_v4_India_All_Years.tif into "
            f"this project folder (or its data/ subfolder), or set the BHI_RASTER_PATH "
            f"environment variable to wherever it lives on your machine."
        )}

    try:
        import rasterio
    except ImportError:
        return {"ok": False, "error": (
            "The 'rasterio' package isn't installed, so the geo-coordinate lookup can't run. "
            "Install it with: pip install rasterio"
        )}

    try:
        import math
        with rasterio.open(raster_path) as src:
            row, col = src.index(lon, lat)
            if row < 0 or col < 0 or row >= src.height or col >= src.width:
                return {"ok": False, "error": f"({lat}, {lon}) falls outside the raster's pixel grid."}
            band_2000 = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
            band_latest = src.read(src.count, window=((row, row + 1), (col, col + 1)))[0, 0]
            if math.isnan(band_2000) or math.isnan(band_latest):
                return {"ok": False, "error": f"No BHI data at ({lat}, {lon}) -- likely a nodata pixel "
                                               f"(open water, etc.)."}
            return {
                "ok": True,
                "bhi_latest": float(band_latest),
                "bhi_2000": float(band_2000),
                "trend_2000_2024": float(band_latest - band_2000),
                "year_latest": YEARS[-1],
            }
    except Exception as e:
        return {"ok": False, "error": f"Raster read failed ({type(e).__name__}: {e})."}


if __name__ == "__main__":
    print(f"Resolved raster path: {resolve_raster_path()}\n")
    samples = {
        "Western Ghats forest (Kerala)": (10.0, 76.8),
        "Indo-Gangetic Plain cropland (Punjab)": (30.9, 75.8),
        "Thar Desert (Rajasthan)": (27.0, 71.0),
        "Sundarbans (West Bengal)": (21.9, 88.9),
        "Delhi urban": (28.6, 77.2),
        "Outside India (sanity check)": (48.85, 2.35),
    }
    for name, (lat, lon) in samples.items():
        print(name, lookup_bhi(lat, lon))