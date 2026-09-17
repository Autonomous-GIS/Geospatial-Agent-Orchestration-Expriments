"""Deterministic format-specific inspectors; artifact contents are never executed.

This module intentionally exposes factual evidence only.  It does not decide how a
workflow should be repaired.  The evidence is designed to support the generic
runtime requirements used by the orchestrator (CRS, geometry, schema, temporal
fitness, spatial coverage, raster resolution/grid alignment, and format
compatibility) without benchmark/task-specific logic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import rasterio
from pyproj import CRS

from gas_server.orchestrator.security.inspection_policy import InspectionPolicy


INSPECTOR_VERSION = "1.2.0"
VECTOR_FORMATS = {"gpkg", "geojson", "shp", "fgb", "kml"}
RASTER_FORMATS = {"tif", "tiff", "geotiff", "cog"}
TABLE_FORMATS = {"csv", "tsv", "parquet", "xlsx"}
HTML_METADATA_PATTERN = re.compile(
    r"<!--\s*EMBEDDED_METADATA:\s*(\{.*?\})\s*-->", re.DOTALL
)


def _schema(frame, max_fields: int) -> dict[str, str]:
    return {
        str(name): str(dtype)
        for name, dtype in list(frame.dtypes.items())[:max_fields]
    }


def _normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _crs_summary(crs_value: Any) -> dict[str, Any]:
    if not crs_value:
        return {
            "crs": None,
            "crs_authority": None,
            "is_projected": False,
            "is_geographic": False,
            "linear_unit": None,
            "linear_unit_to_meter": None,
            "is_metric_crs": False,
        }
    try:
        crs = CRS.from_user_input(crs_value)
        authority = crs.to_authority()
        linear_unit = None
        linear_unit_to_meter = None
        for axis in crs.axis_info:
            unit_name = (axis.unit_name or "").strip()
            if not unit_name:
                continue
            linear_unit = unit_name
            factor = getattr(axis, "unit_conversion_factor", None)
            if factor is not None:
                try:
                    linear_unit_to_meter = float(factor)
                except (TypeError, ValueError):
                    pass
            break
        is_metric = bool(
            crs.is_projected
            and linear_unit
            and linear_unit.lower() in {"meter", "metre", "meters", "metres"}
        )
        return {
            "crs": crs.to_string(),
            "crs_authority": f"{authority[0]}:{authority[1]}" if authority else None,
            "is_projected": bool(crs.is_projected),
            "is_geographic": bool(crs.is_geographic),
            "linear_unit": linear_unit,
            "linear_unit_to_meter": linear_unit_to_meter,
            "is_metric_crs": is_metric,
        }
    except Exception:
        return {
            "crs": str(crs_value),
            "crs_authority": None,
            "is_projected": False,
            "is_geographic": False,
            "linear_unit": None,
            "linear_unit_to_meter": None,
            "is_metric_crs": False,
        }


def _temporal_candidates(frame: pd.DataFrame, max_fields: int = 16) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    column_names = [
        name
        for name in frame.columns
        if any(token in str(name).lower() for token in ("date", "time", "year", "month", "day"))
    ]
    for name in column_names[:max_fields]:
        series = frame[name].dropna()
        if series.empty:
            continue
        lowered = str(name).lower()
        if "year" in lowered:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            numeric = numeric[(numeric >= 1000) & (numeric <= 3000)]
            if not numeric.empty:
                candidates.append(
                    {
                        "field": str(name),
                        "start": str(int(numeric.min())),
                        "end": str(int(numeric.max())),
                        "precision": "year",
                    }
                )
                continue
        parsed = pd.to_datetime(series, errors="coerce", utc=True).dropna()
        if not parsed.empty:
            candidates.append(
                {
                    "field": str(name),
                    "start": parsed.min().isoformat(),
                    "end": parsed.max().isoformat(),
                    "precision": "datetime",
                }
            )
    return candidates


def _temporal_summary(frame: pd.DataFrame) -> dict[str, Any] | None:
    candidates = _temporal_candidates(frame)
    return candidates[0] if candidates else None


def _coordinate_candidates(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return evidence that a table can plausibly be spatialized.

    This is intentionally conservative: names and numeric ranges are reported, but
    no CRS is asserted from coordinate values alone.
    """

    longitude_names = {
        "lon", "lng", "long", "longitude", "x", "xcoord", "xcoordinate", "easting"
    }
    latitude_names = {
        "lat", "latitude", "y", "ycoord", "ycoordinate", "northing"
    }
    normalized = {_normalized_name(name): str(name) for name in frame.columns}
    x_fields = [normalized[name] for name in longitude_names if name in normalized]
    y_fields = [normalized[name] for name in latitude_names if name in normalized]
    results: list[dict[str, Any]] = []
    for x_field in x_fields:
        for y_field in y_fields:
            if x_field == y_field:
                continue
            x = pd.to_numeric(frame[x_field], errors="coerce").dropna()
            y = pd.to_numeric(frame[y_field], errors="coerce").dropna()
            if x.empty or y.empty:
                continue
            geographic_range = bool(
                x.between(-180, 180).all() and y.between(-90, 90).all()
            )
            results.append(
                {
                    "x_field": x_field,
                    "y_field": y_field,
                    "x_min": float(x.min()),
                    "x_max": float(x.max()),
                    "y_min": float(y.min()),
                    "y_max": float(y.max()),
                    "range_compatible_with_lon_lat": geographic_range,
                }
            )
    return results[:12]


def _table_frame(path: Path, format_name: str, max_rows: int) -> tuple[pd.DataFrame, bool]:
    """Read one bounded sample and report whether the sample is the whole table."""

    limit = max(1, int(max_rows))
    if format_name == "csv":
        frame = pd.read_csv(path, nrows=limit + 1)
    elif format_name == "tsv":
        frame = pd.read_csv(path, sep="\t", nrows=limit + 1)
    elif format_name == "parquet":
        frame = pd.read_parquet(path).head(limit + 1)
    else:
        frame = pd.read_excel(path, nrows=limit + 1)
    complete = len(frame) <= limit
    return frame.iloc[:limit].copy(), complete


def _full_temporal_columns_if_needed(
    path: Path,
    format_name: str,
    sampled_frame: pd.DataFrame,
    sample_complete: bool,
) -> tuple[list[dict[str, Any]], bool]:
    """Read only temporal columns through the whole file when a row sample was truncated."""

    sampled = _temporal_candidates(sampled_frame)
    if sample_complete:
        return sampled, True
    temporal_fields = [item["field"] for item in sampled]
    if not temporal_fields:
        return sampled, False
    try:
        if format_name == "csv":
            temporal_frame = pd.read_csv(path, usecols=temporal_fields)
        elif format_name == "tsv":
            temporal_frame = pd.read_csv(path, sep="\t", usecols=temporal_fields)
        elif format_name == "parquet":
            temporal_frame = pd.read_parquet(path, columns=temporal_fields)
        else:
            temporal_frame = pd.read_excel(path, usecols=temporal_fields)
        return _temporal_candidates(temporal_frame), True
    except Exception:
        return sampled, False


def inspect_path(path_value: str | Path, policy: InspectionPolicy) -> dict[str, Any]:
    path = policy.validate_path(path_value)
    format_name = path.suffix.lower().lstrip(".")
    base: dict[str, Any] = {
        "path": str(path),
        "format": format_name,
        "size_bytes": path.stat().st_size,
        "complete": True,
        "limitations": [],
    }

    if format_name in VECTOR_FORMATS:
        frame = gpd.read_file(path)
        crs_summary = _crs_summary(frame.crs)
        if frame.geometry.name in frame.columns:
            attribute_frame = frame.drop(columns=[frame.geometry.name])
        else:
            attribute_frame = frame
        validity = frame.geometry.is_valid if len(frame) else pd.Series([], dtype=bool)
        empty = frame.geometry.is_empty if len(frame) else pd.Series([], dtype=bool)
        null_geometry_count = int(frame.geometry.isna().sum()) if len(frame) else 0
        temporal_candidates = _temporal_candidates(attribute_frame)
        base.update(
            {
                "data_model": "vector",
                "feature_count": int(len(frame)),
                "schema": _schema(attribute_frame, policy.max_schema_fields),
                "schema_fields": [str(item) for item in list(attribute_frame.columns)[: policy.max_schema_fields]],
                "geometry_type": sorted(str(item) for item in frame.geom_type.dropna().unique()),
                "geometry_valid": bool(validity.all()) if len(frame) else True,
                "invalid_geometry_count": int((~validity).sum()) if len(frame) else 0,
                "empty_geometry_count": int(empty.sum()) if len(frame) else 0,
                "null_geometry_count": null_geometry_count,
                "bounds": [float(item) for item in frame.total_bounds] if len(frame) else None,
                "temporal_coverage": temporal_candidates[0] if temporal_candidates else None,
                "temporal_candidates": temporal_candidates,
                "temporal_coverage_complete": True,
                **crs_summary,
            }
        )
        return base

    if format_name in RASTER_FORMATS:
        with rasterio.open(path) as dataset:
            if dataset.count > policy.max_raster_bands:
                base["complete"] = False
                base["limitations"].append("band_count_exceeds_inspection_limit")
            crs_summary = _crs_summary(dataset.crs)
            transform = dataset.transform
            resolution = [abs(float(item)) for item in dataset.res]
            base.update(
                {
                    "data_model": "raster",
                    "width": int(dataset.width),
                    "height": int(dataset.height),
                    "band_count": int(dataset.count),
                    "resolution": resolution,
                    "transform": [float(item) for item in tuple(transform)],
                    "grid_origin": [float(transform.c), float(transform.f)],
                    "bounds": [
                        float(dataset.bounds.left),
                        float(dataset.bounds.bottom),
                        float(dataset.bounds.right),
                        float(dataset.bounds.top),
                    ],
                    "nodata": dataset.nodata,
                    "dtype": str(dataset.dtypes[0]) if dataset.dtypes else None,
                    "driver": dataset.driver,
                    **crs_summary,
                }
            )
        return base

    if format_name in TABLE_FORMATS:
        frame, sample_complete = _table_frame(
            path, format_name, policy.max_features_to_validate
        )
        temporal_candidates, temporal_complete = _full_temporal_columns_if_needed(
            path, format_name, frame, sample_complete
        )
        coordinate_candidates = _coordinate_candidates(frame)
        base.update(
            {
                "data_model": "table",
                "inspected_row_count": int(len(frame)),
                "row_sample_complete": sample_complete,
                "schema": _schema(frame, policy.max_schema_fields),
                "schema_fields": [str(item) for item in list(frame.columns)[: policy.max_schema_fields]],
                "coordinate_candidates": coordinate_candidates,
                "spatializable_from_coordinates": bool(coordinate_candidates),
                "temporal_coverage": temporal_candidates[0] if temporal_candidates else None,
                "temporal_candidates": temporal_candidates,
                "temporal_coverage_complete": temporal_complete,
            }
        )
        # Schema inspection is complete even when row-value inspection is sampled.
        # Preserve that distinction instead of setting the entire evidence record to incomplete.
        if not sample_complete:
            base["limitations"].append("row_values_sampled")
        if not temporal_complete and temporal_candidates:
            base["limitations"].append("temporal_coverage_sampled")
        return base

    if format_name in {"html", "htm"}:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            bounded_text = handle.read(256 * 1024)
        match = HTML_METADATA_PATTERN.search(bounded_text)
        if match is None:
            base.update({"data_model": "map", "complete": False})
            base["limitations"].append("embedded_map_metadata_missing")
            return base
        try:
            metadata = json.loads(match.group(1))
        except (TypeError, json.JSONDecodeError):
            base.update({"data_model": "map", "complete": False})
            base["limitations"].append("embedded_map_metadata_invalid")
            return base
        base.update(
            {
                "data_model": "map",
                "embedded_metadata_present": True,
                "embedded_metadata_keys": sorted(str(item) for item in metadata.keys()),
                "source_artifact": Path(
                    str(metadata.get("source_artifact") or "")
                ).name,
                "mapped_field": metadata.get("mapped_field"),
                "crs": metadata.get("crs"),
                "feature_count": metadata.get("feature_count"),
                "bounds": metadata.get("bbox"),
                "analytical_unit": metadata.get("analytical_unit")
                or metadata.get("spatial_unit"),
                "disclosure": metadata.get("disclosure")
                or metadata.get("assumption_disclosure"),
            }
        )
        return base

    base["data_model"] = "file"
    base["complete"] = False
    base["limitations"].append("unsupported_semantic_inspection_format")
    return base
