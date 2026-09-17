from __future__ import annotations

import os
import random
import re
from pathlib import Path

GIS_LOGICAL_KEYS = {
    "parcels", "parcel", "roads", "road", "highways", "highway", "streets", "street",
    "hospitals", "hospital", "healthcare", "schools", "school", "districts", "district",
    "dem", "elevation", "slope", "aspect", "raster", "contour", "contours",
    "rivers", "river", "streams", "stream", "water", "lakes", "lake",
    "boundary", "boundaries", "municipal", "municipality", "counties", "county", "state",
    "earthquakes", "earthquake", "seismic", "soils", "soil", "parks", "park",
    "railroads", "railroad", "airports", "airport", "buffer", "intersect", "clip",
    "population", "census", "demographics", "tracts", "tract", "data"
}

# Technical geospatial operation words that describe *what was done* to data,
# not *what the data represents*.  Used by extract_subject_hint() to strip
# processing verbs/adjectives from filenames so the pure semantic subject remains.
GEOSPATIAL_OPERATION_WORDS = {
    "projected", "transformed", "reprojected", "clipped", "buffered",
    "merged", "joined", "intersected", "dissolved", "interpolated",
    "rasterized", "vectorized", "converted", "resampled", "subset",
    "cropped", "warped", "mosaicked", "aggregated", "downloaded",
    "fetched", "retrieved", "extracted", "processed", "analyzed",
    "filtered", "selected", "geocoded", "normalized", "standardized",
}


def extract_subject_hint(filename: str) -> str:
    """Extract a clean subject hint from a filename, stripping technical operation words.

    Given a filename like ``centre_county_boundary_projected_123456.gpkg``,
    this returns ``"centre county boundary"`` – the semantic subject without
    processing verbs, agent prefixes, or random tokens.

    Returns an empty string when no meaningful subject can be determined.
    """
    stem = clean_filename_stem(str(filename or ""))
    if not stem:
        return ""
    words = stem.split()
    subject_words = [w for w in words if w.lower() not in GEOSPATIAL_OPERATION_WORDS]
    hint = " ".join(subject_words).strip()
    # Reject low-value single-word hints that carry no semantic meaning
    low_value = {
        "artifact", "output", "dataset", "file", "report", "result",
        "data", "layer", "table", "geospatial",
    }
    if not hint or hint.lower() in low_value:
        return ""
    return hint


def safe_output_stem(
    task: str | None,
    *,
    fallback: str = "result",
    max_words: int = 2,
    max_word_length: int = 12,
) -> str:
    words = re.findall(r"[a-z0-9]+", (task or "").lower())
    selected = [word[:max_word_length] for word in words if word][:max_words]
    if not selected:
        selected = [fallback]
    stem = "_".join(selected).strip("_")
    stem = re.sub(r"_+", "_", stem)
    return stem or fallback


def build_output_filename(
    task: str | None,
    *,
    extension: str,
    fallback: str = "result",
    max_words: int = 2,
) -> str:
    stem = safe_output_stem(
        task,
        fallback=fallback,
        max_words=max_words,
    )
    suffix = f"{random.randint(100000, 999999):06d}"
    if not extension:
        normalized_ext = ""
    else:
        normalized_ext = extension if extension.startswith(".") else f".{extension}"
    return f"{stem}_{suffix}{normalized_ext}"


def build_output_path(
    directory: str,
    task: str | None,
    *,
    extension: str,
    fallback: str = "result",
    max_words: int = 2,
) -> str:
    os.makedirs(directory, exist_ok=True)
    return os.path.join(
        directory,
        build_output_filename(
            task,
            extension=extension,
            fallback=fallback,
            max_words=max_words,
        ),
    )


def clean_filename_stem(filename: str) -> str:
    """Strip agent prefixes, random tokens, hex hashes, and UUIDs to find the core name."""
    stem = Path(str(filename or "")).stem

    # Strip technical operation tokens without deleting later descriptor words.
    # Example: roads_buffered_highways -> roads_highways, not roads.
    ops_pattern = (
        r"(?i)(^|[-_\s])(?:"
        + "|".join(sorted(GEOSPATIAL_OPERATION_WORDS, key=len, reverse=True))
        + r")(?=$|[-_\s])"
    )
    stem = re.sub(ops_pattern, r"\1", stem)
    
    # 1. Strip agent prefixes like pasda_agent-, mapping_agent-, etc.
    stem = re.sub(r"(?i)^[a-z0-9_-]*agent[-_]", "", stem)
    
    # 2. Strip random hashes like -33089-hfoshge-53339
    stem = re.sub(r"(?i)\b\d{4,6}[-_][a-z]{4,8}[-_]\d{4,6}\b", "", stem)
    
    # 3. Strip standard 5-to-8 digit random suffixes like _123456
    stem = re.sub(r"(?i)[-_]\d{5,8}", "", stem)
    
    # 4. Strip hex hashes / UUID-like parts
    stem = re.sub(r"(?i)\b[0-9a-f]{8,}(?:-[0-9a-f]{4,}){0,}\b", "", stem)
    
    # Clean delimiters
    stem = re.sub(r"[_\-\s]+", " ", stem).strip()
    return stem


def extract_naming_metadata(filename: str, task_context: str | None = None) -> dict:
    """Extract semantic logical key, display name, and contextual qualifiers."""
    cleaned_stem = clean_filename_stem(filename)
    words = cleaned_stem.split()
    
    # Identify logical key
    logical_key = None
    geographic_keys = {"county", "counties", "state", "boundary", "boundaries", "data"}
    for word in words:
        w_lower = word.lower()
        if w_lower in GIS_LOGICAL_KEYS and w_lower not in geographic_keys:
            logical_key = w_lower
            break
            
    if not logical_key:
        for word in words:
            w_lower = word.lower()
            if w_lower in GIS_LOGICAL_KEYS:
                logical_key = w_lower
                break
            
    if not logical_key:
        noise = {"pa", "pennsylvania", "county", "state", "result", "output", "dataset", "layer"}
        for word in words:
            if word.lower() not in noise:
                logical_key = word.lower()
                break
        if not logical_key:
            if words:
                logical_key = words[0].lower()
            else:
                logical_key = "dataset"
            
    # Identify qualifiers
    qualifiers = {}
    
    # 1. Check for county/state in words
    for i, word in enumerate(words):
        w_lower = word.lower()
        if w_lower == "county" and i > 0:
            qualifiers["county"] = words[i-1].capitalize()
        elif w_lower == "state" and i > 0:
            qualifiers["state"] = words[i-1].capitalize()
            
    # 2. Look for other potential location/geographic indicators in the stem
    known_locations = {
        "centre": ("county", "Centre"),
        "iowa": ("state", "Iowa"),
        "pa": ("state", "Pennsylvania"),
        "pennsylvania": ("state", "Pennsylvania"),
        "allegheny": ("county", "Allegheny"),
        "lancaster": ("county", "Lancaster"),
        "philadelphia": ("city", "Philadelphia"),
        "pittsburgh": ("city", "Pittsburgh"),
    }
    for word in words:
        w_lower = word.lower()
        if w_lower in known_locations:
            q_type, q_val = known_locations[w_lower]
            qualifiers[q_type] = q_val
            
    # 3. If no county/state found, collect any words that are not the logical key or noise
    if not qualifiers:
        other_words = [w for w in words if w.lower() not in {logical_key, "county", "state", "result", "output", "dataset", "layer", "pa"}]
        if other_words:
            qualifiers["context"] = " ".join(other_words)
            
    # Display name default is title case logical key
    display_name = logical_key.capitalize()
    
    return {
        "logical_key": logical_key,
        "display_name": display_name,
        "qualifiers": qualifiers
    }
