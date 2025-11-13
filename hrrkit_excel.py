#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRRkit – CSV-driven HRR processor (concise IDs, UID in CSV, one-column sources table, plots folder)

CSV schema (in the main directory):
  database.csv    with columns (exact names):
    ID | Scource in IDEEE | Time Unit | Energy Unit | Topic | Filename | ProcessedAt | Peak_kW | Area_kJ
  The script ADDS (if absent) and fills:
    UID | Concise ID

User fills only:
  ID, Scource in IDEEE, Time Unit, Energy Unit, Topic, Filename

Raw files are expected in:
  ./raw_files/<Filename>
On processing, each raw file is MOVED to the per-topic curves_raw with canonical name:
  <concise_id>_(sec|min)_(kw|mw)_raw.csv

Outputs per topic:
  hrr_database/topics/<topic>/curves_clean/<concise_id>.csv
  hrr_database/topics/<topic>/derived/shape_curves/<concise_id>_shape.csv
  hrr_database/topics/<topic>/derived/curve_sources.csv
  hrr_database/topics/<topic>/derived/aggregated_curves.csv
  hrr_database/topics/<topic>/derived/plots/aggregated_curves_plot_qXX.png

Aggregation plot:
  - Input curves in light grey (low opacity)
  - Overlays: mean, max, QXX
  - A one-column, N-rows, borderless “Sources (IDEEE)” table directly under the plot, same width.

Run:
  python hrrkit_excel.py
  # optional:
  python hrrkit_excel.py --quantile 0.95 --base hrr_database --database database.csv --raw raw_files
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

try:  # pragma: no cover - dependency availability differs per environment
    import isbnlib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised when isbnlib missing
    class _IsbnLibFallback:
        """Small subset of isbnlib used by HRRkit.

        The real isbnlib package is preferred, but in environments where it is
        unavailable (such as GitHub Actions runners without optional
        dependencies), this lightweight fallback provides the features required
        by the script:

        * ``canonical`` – keep only digits/X and uppercase the value.
        * ``is_isbn10``/``is_isbn13`` – validate checksum for 10/13-digit ISBNs.
        * ``meta`` – fetch limited metadata from the Google Books API.
        """

        _ISBN10_WEIGHTS = list(range(10, 0, -1))

        @staticmethod
        def canonical(value: str) -> str:
            cleaned = re.sub(r"[^0-9Xx]", "", value or "")
            return cleaned.upper()

        @classmethod
        def _isbn10_checksum(cls, digits: str) -> bool:
            if len(digits) != 10:
                return False
            total = 0
            for weight, char in zip(cls._ISBN10_WEIGHTS, digits):
                if char == "X":
                    val = 10
                elif char.isdigit():
                    val = int(char)
                else:
                    return False
                total += weight * val
            return total % 11 == 0

        @staticmethod
        def _isbn13_checksum(digits: str) -> bool:
            if len(digits) != 13 or not digits.isdigit():
                return False
            total = 0
            for idx, char in enumerate(digits):
                weight = 1 if idx % 2 == 0 else 3
                total += weight * int(char)
            return total % 10 == 0

        @classmethod
        def is_isbn10(cls, value: str) -> bool:
            return cls._isbn10_checksum(cls.canonical(value))

        @classmethod
        def is_isbn13(cls, value: str) -> bool:
            return cls._isbn13_checksum(cls.canonical(value))

        @staticmethod
        def meta(isbn_value: str) -> Dict[str, Any]:
            canonical = _IsbnLibFallback.canonical(isbn_value)
            if len(canonical) not in (10, 13):
                return {}
            url = (
                "https://www.googleapis.com/books/v1/volumes?q=isbn:" + canonical
            )
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                return {}

            items = data.get("items") or []
            if not items:
                return {}
            volume = items[0].get("volumeInfo", {})
            return {
                "Authors": volume.get("authors") or [],
                "Title": volume.get("title", ""),
                "Publisher": volume.get("publisher", ""),
                "Year": volume.get("publishedDate", "")[:4],
            }

    isbnlib = _IsbnLibFallback()


# ----------------------------- Configuration --------------------------------

DEFAULT_BASE_DIR = "hrr_database"
DEFAULT_DATABASE = "database.csv"
DEFAULT_RAW_DIR = "raw_files"
AGGREGATE_STEP_S = 2.0
SHAPE_SUPPORT_POINTS = 25
AGG_QUANTILE = 0.90

TIME_COL_CANDIDATES = ["time_s", "time", "t", "time_raw", "zeit", "Time", "TIME"]
HRR_COL_CANDIDATES  = ["hrr_kW", "hrr", "q", "HRR", "Q", "Leistung", "power"]

REQUIRED_COLUMNS = [
    "ID",
    "Scource in IDEEE",
    "Time Unit",
    "Energy Unit",
    "Topic",
    "Filename",
    "ProcessedAt",
    "Peak_kW",
    "Area_kJ",
]
OPTIONAL_ADDED_COLUMNS = ["UID", "Concise ID"]

TIME_UNIT_MAP = {
    "s": "sec", "sec": "sec", "second": "sec", "seconds": "sec",
    "min": "min", "minute": "min", "minutes": "min",
}
ENERGY_UNIT_MAP = {
    "kw": "kw", "kW": "kw", "KW": "kw",
    "mw": "mw", "MW": "mw",
}
TIME_MULTIPLIER = {"sec": 1.0, "min": 60.0}
POWER_MULTIPLIER = {"kw": 1.0, "mw": 1000.0}


# -------------------------- IEEE source formatting ---------------------------

DOI_ACCEPT_HEADER = {"Accept": "text/x-bibliography; style=ieee"}
DOI_PREFIXES = (
    "doi:", "doi.org/", "https://doi.org/", "http://doi.org/",
    "https://dx.doi.org/", "http://dx.doi.org/",
)
ISBN_PREFIXES = ("isbn:", "isbn-10:", "isbn-13:", "isbn ", "isbn")


def _normalize_doi(doi_value: str) -> str:
    doi_value = (doi_value or "").strip()
    return doi_value.strip().strip(" .;")


def _canonicalize_isbn(isbn_value: str) -> str:
    value = (isbn_value or "").strip()
    if not value:
        return ""
    try:
        canonical = isbnlib.canonical(value)
    except Exception:
        canonical = re.sub(r"[^0-9Xx]", "", value)
    return canonical.upper()


def _is_valid_isbn(isbn_value: str) -> bool:
    if not isbn_value:
        return False
    return isbnlib.is_isbn13(isbn_value) or isbnlib.is_isbn10(isbn_value)


def _extract_source_identifier(value: str) -> Tuple[Optional[str], Optional[str]]:
    raw = (value or "").strip()
    if not raw:
        return None, None

    lower = raw.lower()
    for prefix in DOI_PREFIXES:
        if lower.startswith(prefix):
            candidate = raw[len(prefix):].strip()
            candidate = _normalize_doi(candidate)
            if candidate:
                return "doi", candidate
    if lower.startswith("10.") and "/" in raw:
        candidate = _normalize_doi(raw)
        return "doi", candidate

    for prefix in ISBN_PREFIXES:
        if lower.startswith(prefix):
            candidate = _canonicalize_isbn(raw[len(prefix):])
            if candidate and _is_valid_isbn(candidate):
                return "isbn", candidate

    candidate = _canonicalize_isbn(raw)
    if candidate and _is_valid_isbn(candidate):
        return "isbn", candidate

    return None, None


@lru_cache(maxsize=256)
def _ieee_from_doi(doi_value: str) -> str:
    url = requests.utils.requote_uri(f"https://doi.org/{doi_value}")
    resp = requests.get(url, headers=DOI_ACCEPT_HEADER, timeout=15)
    resp.raise_for_status()
    citation = resp.text.strip()
    if not citation:
        raise ValueError(f"Empty response for DOI '{doi_value}'")
    return citation


@lru_cache(maxsize=256)
def _ieee_from_isbn(isbn_value: str) -> str:
    meta = isbnlib.meta(isbn_value)
    if not meta:
        raise ValueError(f"No metadata returned for ISBN '{isbn_value}'")
    authors = meta.get("Authors", []) or []
    first_author = authors[0] if authors else ""
    title = meta.get("Title", "")
    publisher = meta.get("Publisher", meta.get("PublisherName", ""))
    year = meta.get("Year", "")

    parts: List[str] = []
    if first_author:
        parts.append(first_author.replace(",", ""))
    if title:
        parts.append(title)
    if publisher:
        parts.append(publisher)
    if year:
        parts.append(str(year))

    if not parts:
        raise ValueError(f"Insufficient metadata for ISBN '{isbn_value}'")

    return ", ".join(parts) + "."


def convert_source_to_ieee(source_value: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Return (value, type, identifier, error)."""
    src_type, identifier = _extract_source_identifier(source_value)
    if not src_type or not identifier:
        return source_value, None, None, None

    try:
        if src_type == "doi":
            citation = _ieee_from_doi(identifier)
        else:
            citation = _ieee_from_isbn(identifier)
        return citation, src_type, identifier, None
    except Exception as exc:  # noqa: BLE001 - surface the original reason upstream
        return source_value, src_type, identifier, str(exc)


# ------------------------------ Path helpers --------------------------------

@dataclass
class TopicPaths:
    base_dir: Path
    topics_root: Path
    topic_root: Path
    curves_raw: Path
    curves_clean: Path
    derived: Path
    shape_curves: Path
    plots: Path  # separate folder for plots


def ensure_directory(hrrk_dir: Path) -> None:
    hrrk_dir.mkdir(parents=True, exist_ok=True)


def create_topic_slug(topic_name: str) -> str:
    topic_lower = str(topic_name).strip().lower()
    topic_slug = re.sub(r"[^a-z0-9]+", "_", topic_lower).strip("_")
    return topic_slug or "topic"


def ensure_topic_structure(base_dir: Path, topic_name: str) -> TopicPaths:
    topic_slug = create_topic_slug(topic_name)
    topics_root = base_dir / "topics"
    topic_root = topics_root / topic_slug
    curves_raw = topic_root / "curves_raw"
    curves_clean = topic_root / "curves_clean"
    derived = topic_root / "derived"
    shape_curves = derived / "shape_curves"
    plots = derived / "plots"

    for p in (topics_root, topic_root, curves_raw, curves_clean, derived, shape_curves, plots):
        ensure_directory(p)

    # minimal topic index for traceability
    index_path = base_dir / "hrr_topics_index.csv"
    existing_topics: List[str] = []
    legacy_format = True
    if index_path.exists():
        legacy_format = False
        with index_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or len(header) != 1 or header[0].strip().lower() != "topic_name":
                legacy_format = True
            for row in reader:
                if not row:
                    continue
                entry = (row[-1] or "").strip()
                if entry and entry not in existing_topics:
                    existing_topics.append(entry)

    topic_entry = (str(topic_name).strip() or topic_slug)
    needs_write = legacy_format or not index_path.exists()
    if topic_entry and topic_entry not in existing_topics:
        existing_topics.append(topic_entry)
        needs_write = True

    if needs_write:
        with index_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["topic_name"])
            for entry in existing_topics:
                writer.writerow([entry])

    return TopicPaths(
        base_dir=base_dir,
        topics_root=topics_root,
        topic_root=topic_root,
        curves_raw=curves_raw,
        curves_clean=curves_clean,
        derived=derived,
        shape_curves=shape_curves,
        plots=plots,
    )


# --------------------------- CSV / units helpers ----------------------------

def normalize_time_unit(unit_in: str) -> str:
    key = (unit_in or "").strip()
    return TIME_UNIT_MAP.get(key.lower(), "")


def normalize_energy_unit(unit_in: str) -> str:
    key = (unit_in or "").strip()
    return ENERGY_UNIT_MAP.get(key, ENERGY_UNIT_MAP.get(key.lower(), ""))


def _ensure_string_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Ensure columns exist and are of pandas 'string' dtype to avoid dtype FutureWarnings."""
    for c in columns:
        if c not in df.columns:
            df[c] = pd.Series(dtype="string")
        else:
            if str(df[c].dtype) != "string":
                df[c] = df[c].astype("string")
    return df


def read_database(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        # Create template with proper dtypes
        template_cols = {c: pd.Series(dtype="string") for c in REQUIRED_COLUMNS + OPTIONAL_ADDED_COLUMNS}
        template_cols["Peak_kW"] = pd.Series(dtype="float64")
        template_cols["Area_kJ"] = pd.Series(dtype="float64")
        empty_df = pd.DataFrame(template_cols)
        empty_df.to_csv(csv_path, index=False)
        print(f"Created template CSV at: {csv_path}")
        return empty_df

    try:
        db_df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        template_cols = {c: pd.Series(dtype="string") for c in REQUIRED_COLUMNS + OPTIONAL_ADDED_COLUMNS}
        template_cols["Peak_kW"] = pd.Series(dtype="float64")
        template_cols["Area_kJ"] = pd.Series(dtype="float64")
        empty_df = pd.DataFrame(template_cols)
        empty_df.to_csv(csv_path, index=False)
        print(f"Recreated empty CSV at: {csv_path}")
        return empty_df

    # Ensure required/optional columns exist
    for col in REQUIRED_COLUMNS + OPTIONAL_ADDED_COLUMNS:
        if col not in db_df.columns:
            db_df[col] = np.nan

    # Enforce string dtype for string columns
    string_cols = [
        "ID", "Scource in IDEEE", "Time Unit", "Energy Unit", "Topic", "Filename",
        "ProcessedAt", "UID", "Concise ID"
    ]
    db_df = _ensure_string_columns(db_df, string_cols)

    # Ensure numeric types for metrics
    for num_col in ["Peak_kW", "Area_kJ"]:
        if num_col not in db_df.columns:
            db_df[num_col] = pd.Series(dtype="float64")
        else:
            db_df[num_col] = pd.to_numeric(db_df[num_col], errors="coerce")

    # Stable order
    ordered = [c for c in REQUIRED_COLUMNS + OPTIONAL_ADDED_COLUMNS if c in db_df.columns]
    ordered += [c for c in db_df.columns if c not in ordered]
    db_df = db_df[ordered]
    return db_df


def write_database(csv_path: Path, db_df: pd.DataFrame) -> None:
    db_df.to_csv(csv_path, index=False)


# ------------------------------ CSV utilities --------------------------------

def _looks_like_numeric_header(value: str) -> bool:
    if value is None:
        return False
    return bool(re.fullmatch(r"\s*[-+]?\d+(?:[.,]\d+)?\s*", str(value)))


def _frame_contains_char(df: pd.DataFrame, char: str) -> bool:
    obj = df.select_dtypes(include=["object", "string"])
    if obj.empty:
        return False
    return obj.apply(lambda c: c.astype(str).str.contains(char, na=False)).any().any()


def load_raw_curve(path: Path) -> pd.DataFrame:
    """Robust CSV reader (auto separator; supports decimal commas)."""

    def read_with(sep: Optional[str], header: Optional[int]) -> pd.DataFrame:
        kwargs = {"engine": "python"}
        if sep is not None:
            kwargs["sep"] = sep
        return pd.read_csv(path, header=header, **kwargs)

    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        sample = ""

    candidate_separators: List[Optional[str]] = [None]
    for sep in (";", "\t", "|"):
        if sep in sample:
            candidate_separators.append(sep)

    last_exc: Optional[Exception] = None
    seen: set[Optional[str]] = set()

    for sep in candidate_separators:
        if sep in seen:
            continue
        seen.add(sep)

        try:
            df = read_with(sep, header=0)
        except Exception as exc:
            last_exc = exc
            continue

        if df.empty:
            continue

        if sep is None and _frame_contains_char(df.head(5), ";"):
            continue

        if sep is not None and df.shape[1] == 1:
            continue

        if all(_looks_like_numeric_header(col) for col in df.columns):
            try:
                df = read_with(sep, header=None)
            except Exception as exc:
                last_exc = exc
                continue

        return df

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        last_exc = exc
    else:
        if not df.empty:
            if all(_looks_like_numeric_header(col) for col in df.columns):
                df = pd.read_csv(path, header=None)
            return df

    if last_exc is not None:
        raise last_exc
    raise ValueError("CSV file contains no data")


def detect_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lowered: Dict[str, str] = {}
    for col in columns:
        lowered[str(col).strip().lower()] = col
    for cand in candidates:
        key = cand.strip().lower()
        if key in lowered:
            return lowered[key]
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(",", ".", regex=False)
    cleaned = cleaned.str.replace(r"\s+", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


# ------------------------- Concise ID / filenames ----------------------------

def _slugify(text_in: str) -> str:
    txt = re.sub(r"[^A-Za-z0-9]+", "_", (text_in or "").strip())
    return txt.strip("_")


def make_concise_id_and_uid(id_str: str, ideee: str, max_total_len: int = 40) -> Tuple[str, str]:
    """
    Returns (concise_id, uid6). UID is 6 hex chars from SHA1 over "ID|IDEEE".
    concise_id pattern: <idslug>_<ideeeshort>_<uid6> (≤ max_total_len)
    """
    idslug = _slugify(id_str)[:16] or "id"
    tokens = [t for t in _slugify(ideee).split("_") if t]
    if tokens:
        ideeeshort = ""
        for tok in tokens[:2]:
            candidate = (ideeeshort + ("_" if ideeeshort else "") + tok)
            if len(candidate) <= 14:
                ideeeshort = candidate
            else:
                break
        ideeeshort = ideeeshort or tokens[0][:14]
    else:
        ideeeshort = "src"

    uid6 = hashlib.sha1(f"{id_str}|{ideee}".encode("utf-8")).hexdigest()[:6]
    base = f"{idslug}_{ideeeshort}_{uid6}"
    if len(base) <= max_total_len:
        return base, uid6

    # Trim ideeeshort, then idslug if needed
    overflow = len(base) - max_total_len
    if overflow > 0 and len(ideeeshort) > 3:
        ideeeshort = ideeeshort[:-min(overflow, len(ideeeshort) - 3)]
    base = f"{idslug}_{ideeeshort}_{uid6}"
    if len(base) <= max_total_len:
        return base, uid6

    overflow = len(base) - max_total_len
    if overflow > 0 and len(idslug) > 3:
        idslug = idslug[:-min(overflow, len(idslug) - 3)]
    base = f"{idslug}_{ideeeshort}_{uid6}"
    return base[:max_total_len], uid6


def ensure_unique_concise_id(
    base_concise_id: str,
    paths: TopicPaths,
    time_unit_norm: str,
    energy_unit_norm: str,
) -> Tuple[str, bool]:
    """Return a concise_id that does not collide with existing files."""

    candidate = base_concise_id
    counter = 2
    def _has_conflict(name: str) -> bool:
        raw_name = paths.curves_raw / f"{name}_{time_unit_norm}_{energy_unit_norm}_raw.csv"
        clean_name = paths.curves_clean / f"{name}.csv"
        shape_name = paths.shape_curves / f"{name}_shape.csv"
        return raw_name.exists() or clean_name.exists() or shape_name.exists()

    while _has_conflict(candidate):
        candidate = f"{base_concise_id}_v{counter:02d}"
        counter += 1
    return candidate, candidate != base_concise_id


# ----------------------------- Curve processing ------------------------------

def normalise_curve_seconds_kw(
    df: pd.DataFrame,
    time_column: str,
    hrr_column: str,
    time_unit: str,
    energy_unit: str,
) -> pd.DataFrame:
    """Convert time to seconds and HRR to kW based on database units."""
    time_unit_norm = normalize_time_unit(time_unit)
    energy_unit_norm = normalize_energy_unit(energy_unit)
    if not time_unit_norm or not energy_unit_norm:
        raise ValueError(f"Unsupported units: time='{time_unit}' energy='{energy_unit}'")

    t_mul = TIME_MULTIPLIER[time_unit_norm]
    p_mul = POWER_MULTIPLIER[energy_unit_norm]

    time_series = coerce_numeric(df[time_column]) * t_mul
    hrr_series  = coerce_numeric(df[hrr_column]) * p_mul

    mask = time_series.notna() & hrr_series.notna()
    cleaned = pd.DataFrame(
        {"time_s": time_series[mask].to_numpy(dtype=float),
         "hrr_kW": hrr_series[mask].to_numpy(dtype=float)}
    ).sort_values("time_s")

    if cleaned.empty:
        raise ValueError("Curve does not contain numeric time/HRR data")

    cleaned["hrr_kW"] = cleaned["hrr_kW"].clip(lower=0.0)
    cleaned["time_s"] -= cleaned["time_s"].iloc[0]

    order = np.argsort(cleaned["time_s"].to_numpy())
    cleaned = cleaned.iloc[order].reset_index(drop=True)
    if len(cleaned) > 1:
        uniq_mask = np.r_[True, np.diff(cleaned["time_s"].to_numpy()) > 0]
        cleaned = cleaned.loc[uniq_mask].reset_index(drop=True)
    return cleaned


def reduce_curve_to_shape(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return curve.copy()
    times = curve["time_s"].to_numpy(dtype=float)
    hrr = curve["hrr_kW"].to_numpy(dtype=float)

    if len(times) <= SHAPE_SUPPORT_POINTS:
        return curve.copy()

    target_times = np.linspace(times.min(), times.max(), SHAPE_SUPPORT_POINTS)
    target_hrr = np.interp(target_times, times, hrr)
    return pd.DataFrame({"time_s": target_times, "hrr_kW": target_hrr})


def integrate_kW_over_s_to_kJ(t_s: np.ndarray, q_kW: np.ndarray) -> float:
    try:
        return float(np.trapezoid(q_kW, t_s))  # NumPy ≥ 2.0
    except AttributeError:
        return float(np.trapz(q_kW, t_s))      # Fallback (older NumPy)


def curve_stats(clean_df: pd.DataFrame) -> Dict[str, float]:
    t = clean_df["time_s"].to_numpy(dtype=float)
    q = clean_df["hrr_kW"].to_numpy(dtype=float)
    if t.size == 0:
        return {"Peak_kW": np.nan, "Area_kJ": np.nan}
    peak = float(np.nanmax(q))
    area_kJ = integrate_kW_over_s_to_kJ(t, q)
    return {"Peak_kW": peak, "Area_kJ": area_kJ}


# ------------------------ Curve source registry (per topic) -------------------

def sources_registry_path(paths: TopicPaths) -> Path:
    return paths.derived / "curve_sources.csv"


def load_sources_registry(paths: TopicPaths) -> pd.DataFrame:
    p = sources_registry_path(paths)
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "concise_id", "UID", "ID_input", "IDEEE", "original_filename", "canonical_raw_name", "processed_at"
    ])


def save_sources_registry(paths: TopicPaths, registry_df: pd.DataFrame) -> None:
    registry_df = registry_df.drop_duplicates(subset=["concise_id"], keep="last")
    registry_df.to_csv(sources_registry_path(paths), index=False)


# ---------------------- Row processing (move + clean + shape) ----------------

def process_row_to_files(
    row: pd.Series,
    raw_dir: Path,
    base_dir: Path,
    now_str: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Returns:
      (stats or None, error or None, topic or None, meta for registry/database or None)
    """
    row_id = str(row["ID"]).strip()
    ideee_input = str(row["Scource in IDEEE"]).strip()
    topic = str(row["Topic"]).strip()
    time_unit = str(row["Time Unit"]).strip()
    energy_unit = str(row["Energy Unit"]).strip()
    filename = str(row["Filename"]).strip()

    if not (row_id and topic and time_unit and energy_unit and filename):
        return None, "Missing required fields", None, None

    concise_id, uid6 = make_concise_id_and_uid(row_id, ideee_input)
    ideee_formatted, src_type, identifier, conv_error = convert_source_to_ieee(ideee_input)
    if src_type:
        if conv_error:
            print(f"    [warning] Could not convert {src_type.upper()} '{identifier}': {conv_error}")
        else:
            print(f"    Source interpreted as {src_type.upper()} '{identifier}' → IEEE reference.")
    ideee = ideee_formatted
    time_unit_norm = normalize_time_unit(time_unit)
    energy_unit_norm = normalize_energy_unit(energy_unit)
    if not time_unit_norm or not energy_unit_norm:
        return None, f"Unsupported units: time='{time_unit}' energy='{energy_unit}'", None, None

    raw_src_path = (raw_dir / filename).resolve()
    if not raw_src_path.exists():
        return None, f"Raw file not found: {raw_src_path}", None, None

    # Prepare topic directories
    paths = ensure_topic_structure(base_dir, topic)

    # Ensure we do not overwrite previous submissions (same ID/IDEEE)
    concise_id, renamed = ensure_unique_concise_id(concise_id, paths, time_unit_norm, energy_unit_norm)
    if renamed:
        print(f"    Concise ID already present, storing as '{concise_id}'.")

    # Move raw → curves_raw with canonical name
    canonical_raw_name = f"{concise_id}_{time_unit_norm}_{energy_unit_norm}_raw.csv"
    raw_dst_path = (paths.curves_raw / canonical_raw_name).resolve()
    try:
        ensure_directory(paths.curves_raw)
        shutil.move(str(raw_src_path), str(raw_dst_path))
    except Exception as ex:
        return None, f"Failed moving raw file: {ex}", None, None

    # Load moved raw and process
    df_raw = load_raw_curve(raw_dst_path)
    if df_raw.shape[1] < 2:
        return None, "Raw CSV must have at least 2 columns", None, None
    time_col = detect_column(df_raw.columns, TIME_COL_CANDIDATES) or df_raw.columns[0]
    hrr_col  = detect_column(df_raw.columns, HRR_COL_CANDIDATES)  or df_raw.columns[1]

    clean_df = normalise_curve_seconds_kw(df_raw, time_col, hrr_col, time_unit, energy_unit)
    shape_df = reduce_curve_to_shape(clean_df)

    # Save cleaned + shape (overwrite)
    clean_path = paths.curves_clean / f"{concise_id}.csv"
    shape_path = paths.shape_curves / f"{concise_id}_shape.csv"
    clean_df.to_csv(clean_path, index=False)
    shape_df.to_csv(shape_path, index=False)

    stats = curve_stats(clean_df)
    print(f"  ✓ {canonical_raw_name} → {clean_path.name} (+shape)")

    meta = {
        "concise_id": concise_id,
        "UID": uid6,
        "ID_input": row_id,
        "IDEEE": ideee,
        "original_filename": filename,
        "canonical_raw_name": canonical_raw_name,
        "processed_at": now_str,
        "topic": topic,
    }
    return stats, None, topic, meta


# ------------------------------ Aggregation ----------------------------------

def _extract_concise_id_from_curve_filename(p: Path) -> str:
    nm = p.name
    if nm.endswith("_shape.csv"):
        return nm[:-len("_shape.csv")]
    if nm.endswith(".csv"):
        return nm[:-4]
    return nm


def _compute_sources_table_height(n_rows: int, font_size: int = 8) -> float:
    """
    Figure-fraction height reserved for the one-column sources table.
    Scales with number of sources, capped so the plot still has room.
    """
    if n_rows <= 0:
        return 0.12
    per_row = 0.03 if font_size <= 9 else 0.035
    base = 0.12
    return min(0.48, base + per_row * n_rows)


def _add_sources_table_below(fig, ax, sources: List[str], table_height: float, font_size: int = 8,
                             header: str = "Sources (IDEEE)") -> None:
    """
    One-column, N-rows, borderless table directly UNDER the plot.
    - Header sits a bit lower.
    - First characters of rows align with the 'S' of the header.
    - Wrapping width scales to the plot's actual pixel width (full-width usage).
    """
    # Clean + unique
    cleaned: List[str] = []
    seen_loc: Set[str] = set()
    for s in sources:
        s = str(s).strip()
        if s and s not in seen_loc:
            cleaned.append(s)
            seen_loc.add(s)
    if not cleaned:
        cleaned = ["(no sources)"]

    # Compute wrapping width (characters) from the plot width in pixels
    pos = ax.get_position()  # figure fractions
    plot_pix_width = pos.width * fig.bbox.width
    approx_char_px = max(1.0, font_size * 0.60)  # ~0.6em per char
    wrap_chars = max(20, int(plot_pix_width / approx_char_px) - 4)

    wrapped = [textwrap.fill(x, width=wrap_chars) for x in cleaned]
    cell_text = [[w] for w in wrapped]  # 1 column, N rows

    # New axes directly below the plot, same width
    x0, y0, w, _ = pos.x0, pos.y0, pos.width, pos.height
    table_y0 = max(0.01, y0 - table_height)
    table_ax = fig.add_axes([x0, table_y0, w, table_height])
    table_ax.axis("off")

    # Header a bit lower; left aligned at x=0 so rows start parallel to 'S'
    table_ax.text(0.0, 0.90, header, ha="left", va="top",
                  fontsize=10, fontweight="bold", transform=table_ax.transAxes)

    # Borderless table occupying full width under the header
    tbl = table_ax.table(
        cellText=cell_text,
        colLabels=None,
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        bbox=[0.0, 0.0, 1.0, 0.86],  # full width under the header
        colWidths=[1.0],
    )

    # Remove borders/padding so text aligns flush with header start
    for cell in tbl.get_celld().values():
        cell.set_edgecolor((0, 0, 0, 0))
        cell.set_facecolor((1, 1, 1, 0))
        cell.set_linewidth(0)
        try:
            cell.PAD = 0.0  # remove left/right padding if available
        except Exception:
            pass
        cell.set_text_props(fontsize=font_size, ha="left", va="top")


def aggregate_topic(base_dir: Path, topic: str, quantile_q: float = AGG_QUANTILE) -> None:
    paths = ensure_topic_structure(base_dir, topic)
    shape_files = sorted(paths.shape_curves.glob("*.csv"))
    if shape_files:
        source_files = shape_files
        print(f"  [{topic}] Using shape curves for aggregation ({len(source_files)}).")
    else:
        source_files = sorted(paths.curves_clean.glob("*.csv"))
        print(f"  [{topic}] Using cleaned curves for aggregation ({len(source_files)}).")

    if not source_files:
        print(f"  [{topic}] No curves to aggregate.")
        return

    # Load sources registry (for caption/table)
    registry = load_sources_registry(paths)
    reg_map = {str(r["concise_id"]): str(r["IDEEE"]) for _, r in registry.iterrows()}

    grids_list: List[np.ndarray] = []
    values_list: List[np.ndarray] = []
    used_sources: List[str] = []
    used_concise_ids: List[str] = []
    max_time_detected = 0.0

    for curve_file in source_files:
        concise_id = _extract_concise_id_from_curve_filename(curve_file)
        used_concise_ids.append(concise_id)
        ideee = reg_map.get(concise_id, "")
        if ideee:
            used_sources.append(ideee)

        df_in = pd.read_csv(curve_file)
        val_col = None
        for cand in ("hrr_kW", "shape", "hrr_norm", "q_kW", "HRR"):
            if cand in df_in.columns:
                val_col = cand
                break
        if "time_s" not in df_in.columns or val_col is None:
            print(f"    Skipping '{curve_file.name}' (missing columns).")
            continue

        t_arr = pd.to_numeric(df_in["time_s"], errors="coerce").to_numpy()
        v_arr = pd.to_numeric(df_in[val_col], errors="coerce").to_numpy()
        ok = np.isfinite(t_arr) & np.isfinite(v_arr)
        t_arr, v_arr = t_arr[ok], v_arr[ok]
        if t_arr.size < 2:
            print(f"    Skipping '{curve_file.name}' (too few points).")
            continue
        order = np.argsort(t_arr)
        t_arr, v_arr = t_arr[order], v_arr[order]
        uniq_mask = np.r_[True, np.diff(t_arr) > 0]
        t_arr, v_arr = t_arr[uniq_mask], v_arr[uniq_mask]
        if t_arr.size < 2:
            print(f"    Skipping '{curve_file.name}' (degenerate after cleaning).")
            continue

        max_time_detected = max(max_time_detected, float(t_arr.max()))
        grids_list.append(t_arr)
        values_list.append(v_arr)

    if not grids_list:
        print(f"  [{topic}] No valid curves to aggregate after filtering.")
        return

    step = AGGREGATE_STEP_S
    agg_grid = np.arange(0.0, max_time_detected + step, step)

    interp_samples: List[np.ndarray] = []
    for t_arr, v_arr in zip(grids_list, values_list):
        interp_samples.append(np.interp(agg_grid, t_arr, v_arr, left=np.nan, right=np.nan))

    stack = np.vstack(interp_samples)  # [n_curves, n_times]
    keep = np.sum(np.isfinite(stack), axis=0) > 0
    if not np.any(keep):
        print(f"  [{topic}] No overlapping time support across curves.")
        return

    grid_kept = agg_grid[keep]
    stack_kept = stack[:, keep]

    mean_curve = np.nanmean(stack_kept, axis=0)
    max_curve  = np.nanmax(stack_kept, axis=0)
    quant_curve = np.nanquantile(stack_kept, quantile_q, axis=0)

    aggregate_df = pd.DataFrame(
        {
            "time_s": grid_kept,
            "mean_hrr_kW": mean_curve,
            "max_hrr_kW": max_curve,
            f"q{int(quantile_q*100)}_hrr_kW": quant_curve,
        }
    )

    paths.derived.mkdir(parents=True, exist_ok=True)
    destination_csv = paths.derived / "aggregated_curves.csv"
    aggregate_df.to_csv(destination_csv, index=False)

    # Unique sources (fallback to concise IDs if registry empty)
    unique_sources: List[str] = []
    seen_src: Set[str] = set()
    for s in used_sources:
        s_clean = str(s).strip()
        if s_clean and s_clean not in seen_src:
            unique_sources.append(s_clean)
            seen_src.add(s_clean)
    if not unique_sources:
        unique_sources = used_concise_ids

    # Compute required bottom margin from number of sources (single-column table)
    table_h = _compute_sources_table_height(len(unique_sources), font_size=8)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    # Reserve vertical space at bottom for the sources table
    fig.subplots_adjust(bottom=0.08 + table_h)

    # Plot inputs in light grey + overlays
    for t_arr, v_arr in zip(grids_list, values_list):
        ax.plot(t_arr, v_arr, color="0.7", linewidth=0.8, alpha=0.2)
    ax.plot(grid_kept, mean_curve, label="Mean", linewidth=2.0)
    ax.plot(grid_kept, max_curve,  label="Max",  linewidth=1.8, linestyle="--")
    ax.plot(grid_kept, quant_curve, label=f"Q{int(quantile_q*100)}", linewidth=1.8, linestyle=":")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("HRR [kW]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")

    # Add 1-column, N-rows, borderless table under the plot (same width)
    _add_sources_table_below(fig, ax, unique_sources, table_h, font_size=8)

    # Save to separate plots folder
    destination_png = paths.plots / f"aggregated_curves_plot_q{int(quantile_q*100)}.png"
    fig.savefig(destination_png, bbox_inches="tight")
    plt.close(fig)

    print(f"  [{topic}] Aggregated curves saved -> {destination_csv}")
    print(f"  [{topic}] Aggregated plot  saved -> {destination_png}")


# ----------------------------------- Main ------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CSV-driven HRR processor (concise IDs, UID in CSV, one-column sources table, plots folder).")
    parser.add_argument(
        "--database", "--csv", "--excel", dest="database", default=DEFAULT_DATABASE,
        help="Database CSV file (default: database.csv)"
    )
    parser.add_argument("--raw",   default=DEFAULT_RAW_DIR, help="Folder with raw CSV files (default: raw_files)")
    parser.add_argument("--base",  default=DEFAULT_BASE_DIR, help="Output base directory (default: hrr_database)")
    parser.add_argument("--quantile", type=float, default=AGG_QUANTILE, help="Quantile for aggregate curve (0..1)")
    args = parser.parse_args()

    database_path = Path(args.database).resolve()
    raw_dir = Path(args.raw).resolve()
    base_dir = Path(args.base).resolve()

    ensure_directory(raw_dir)
    ensure_directory(base_dir)

    db_df = read_database(database_path)
    if db_df.empty:
        print("No rows to process. Fill the template and rerun.")
        return

    # Make sure string columns truly are string dtype (prevents FutureWarning on assignment)
    db_df = _ensure_string_columns(db_df, [
        "ID", "Scource in IDEEE", "Time Unit", "Energy Unit", "Topic", "Filename",
        "ProcessedAt", "UID", "Concise ID"
    ])

    processed_topics: Set[str] = set()
    registry_updates: Dict[str, List[Dict[str, Any]]] = {}

    now_str = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    for idx, row in db_df.iterrows():
        mandatory_ok = all(
            str(row[col]).strip() not in ("", "nan", "None")
            for col in ("ID", "Scource in IDEEE", "Time Unit", "Energy Unit", "Topic", "Filename")
        )
        if not mandatory_ok:
            continue

        stats, error, topic, meta = process_row_to_files(row, raw_dir=raw_dir, base_dir=base_dir, now_str=now_str)
        if error:
            print(f"[row {idx}] {error}")
            continue

        # Safe assignments (string columns are 'string' dtype)
        db_df.at[idx, "ProcessedAt"] = pd.Series([now_str], dtype="string").iloc[0]
        db_df.at[idx, "UID"] = pd.Series([meta["UID"]], dtype="string").iloc[0]
        db_df.at[idx, "Concise ID"] = pd.Series([meta["concise_id"]], dtype="string").iloc[0]
        db_df.at[idx, "Scource in IDEEE"] = pd.Series([meta["IDEEE"]], dtype="string").iloc[0]
        db_df.at[idx, "Peak_kW"] = float(stats["Peak_kW"])
        db_df.at[idx, "Area_kJ"]  = float(stats["Area_kJ"])

        processed_topics.add(topic)
        registry_updates.setdefault(topic, []).append(meta)

    # Write updated database
    write_database(database_path, db_df)
    print(f"Database CSV updated -> {database_path}")

    # Update per-topic registry
    for topic, updates in registry_updates.items():
        paths = ensure_topic_structure(base_dir, topic)
        existing = load_sources_registry(paths)
        upd_df = pd.DataFrame(updates)[[
            "concise_id", "UID", "ID_input", "IDEEE", "original_filename", "canonical_raw_name", "processed_at"
        ]]
        merged = pd.concat([existing, upd_df], ignore_index=True)
        save_sources_registry(paths, merged)

    # Aggregate per processed topic
    for topic in sorted(processed_topics):
        aggregate_topic(base_dir, topic, quantile_q=args.quantile)

    print("Done.")


if __name__ == "__main__":
    main()
