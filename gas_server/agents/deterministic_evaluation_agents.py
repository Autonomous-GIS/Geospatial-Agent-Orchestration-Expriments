"""Deterministic GAS evaluation worker agents (A1-A8).

Zero-LLM, purely deterministic implementations of the 8 evaluation worker agents
for the GIS-aware orchestrator benchmark:
- A1: Data Retrieval Agent (data_retrieval_agent)
- A2: Vector Analysis A (vector_analysis_a)
- A3: Vector Analysis B (vector_analysis_b)
- A4: Raster Analysis Agent (raster_analysis_agent)
- A5: Projection Agent (projection_agent)
- A6: Conversion Agent (conversion_agent)
- A7: Statistics Agent (statistics_agent)
- A8: Mapping Agent (mapping_agent)
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, ClassVar

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.warp
from rasterio.enums import Resampling
from shapely.geometry import Point, mapping, shape
from shapely.validation import make_valid

from gas_server.core.config import DATA_DIR
from gas_server.core.geo_agent import GeoAgent, ProgressCallback


# ==============================================================================
# Helper functions
# ==============================================================================

def _extract_target_crs(query: str, parameters: dict[str, Any]) -> str | None:
    """Extract EPSG or CRS string from parameters or natural-language query."""
    if parameters.get("target_crs"):
        return str(parameters["target_crs"]).strip()
    if parameters.get("crs"):
        return str(parameters["crs"]).strip()
    if parameters.get("source_crs"):
        return str(parameters["source_crs"]).strip()

    # Search query text for EPSG codes
    match = re.search(r"\bEPSG[:\s_]?(\d{4,5})\b", query, re.IGNORECASE)
    if match:
        return f"EPSG:{match.group(1)}"
    if "wgs84" in query.lower() or "wgs 84" in query.lower() or "4326" in query:
        return "EPSG:4326"
    if "nad83" in query.lower() and "utm" in query.lower() and "18" in query:
        return "EPSG:26918"
    if "26918" in query:
        return "EPSG:26918"
    if "3857" in query or "web mercator" in query.lower():
        return "EPSG:3857"
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _first_number(value: Any, default: float) -> float:
    for item in _as_list(value):
        try:
            return float(item)
        except (TypeError, ValueError):
            continue
    return float(default)


def _extract_distance_meters(query: str, parameters: dict[str, Any]) -> float | None:
    """Extract metric distance in meters from parameters or natural-language query."""
    if parameters.get("distance_meters") is not None:
        try:
            return float(parameters["distance_meters"])
        except (ValueError, TypeError):
            pass
    if parameters.get("distance") is not None:
        try:
            val = float(parameters["distance"])
            unit = str(parameters.get("unit", "meters")).lower()
            if "km" in unit or "kilometer" in unit:
                return val * 1000.0
            return val
        except (ValueError, TypeError):
            pass

    # Search query text
    km_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|kilometer|kilometre|kilometers)", query, re.IGNORECASE)
    if km_match:
        return float(km_match.group(1)) * 1000.0

    m_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meter|metre|meters)", query, re.IGNORECASE)
    if m_match:
        return float(m_match.group(1))

    return None


# ==============================================================================
# A1: Data Retrieval Agent
# ==============================================================================

class ControlledDataRetrievalAgent(GeoAgent):
    """Retrieves registered benchmark datasets from the configured catalog."""

    agent_id: ClassVar[str] = "data_retrieval_agent"
    agent_name: ClassVar[str] = "Data Retrieval Agent"
    agent_version: ClassVar[str] = "2.0.0"
    agent_description: ClassVar[str] = (
        "Retrieves registered benchmark datasets and returns the selected vector, "
        "raster, or tabular artifact. The service does not determine downstream analytical fitness."
    )
    requires_input_datasets: ClassVar[bool] = False
    requires_model_credentials: ClassVar[bool] = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(output_dir=DATA_DIR / self.agent_id, **kwargs)

    def run(
        self,
        query: str,
        input_dataset_paths: list[str] | str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        params = dict(self.request_parameters or {})
        exclude_ids = set(params.get("exclude_artifact_ids") or params.get("exclude_ids") or [])

        # Also check query for explicit exclusion patterns
        for token in re.findall(r"\bfx_[a-z0-9_]+\b", query, re.IGNORECASE):
            if "exclude" in query.lower() or "other" in query.lower() or "another" in query.lower():
                exclude_ids.add(token.lower())

        # Load active catalog from environment or parameter
        catalog_path = os.getenv("GAS_BENCHMARK_CATALOG") or params.get("catalog_path")
        catalog_entries: list[dict[str, Any]] = []

        if catalog_path and Path(catalog_path).exists():
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog_data = json.load(f)
                catalog_entries = catalog_data if isinstance(catalog_data, list) else catalog_data.get("entries", [])
        else:
            # Fallback catalog lookup in benchmark/fixtures
            fixtures_dir = Path(__file__).resolve().parents[2] / "benchmark" / "fixtures"
            manifest_file = fixtures_dir / "FIXTURE_MANIFEST.json"
            if manifest_file.exists():
                with open(manifest_file, "r", encoding="utf-8") as f:
                    catalog_entries = json.load(f)

        requested_format = str(params.get("format") or params.get("file_format") or "").lower().lstrip(".")
        requested_years = {str(item) for item in _as_list(params.get("year") or params.get("years")) if str(item).strip()}
        requested_artifacts = {
            str(item).lower()
            for key in ("artifact_id", "target_artifact", "dataset_id", "file_name", "filename")
            for item in _as_list(params.get(key))
            if str(item).strip()
        }
        retrieval_text = " ".join(
            [
                query,
                json.dumps(params, sort_keys=True, default=str),
            ]
        ).lower()
        if not requested_years:
            requested_years = set(re.findall(r"\b(?:19|20)\d{2}\b", retrieval_text))

        def _entry_text(entry: dict[str, Any]) -> str:
            return " ".join(
                str(entry.get(key) or "")
                for key in ("artifact_id", "id", "file_name", "filename", "file_path", "format", "description", "tags")
            ).lower()

        def _score_entry(entry: dict[str, Any]) -> int:
            entry_id = str(entry.get("artifact_id") or entry.get("id") or "").lower()
            file_name = str(entry.get("file_name") or entry.get("filename") or "").lower()
            entry_format = str(entry.get("format") or Path(file_name).suffix.lstrip(".")).lower()
            text = _entry_text(entry)
            score = 0

            if requested_artifacts:
                if any(item == entry_id or item == file_name or item in text for item in requested_artifacts):
                    score += 1000
                else:
                    score -= 500
            if requested_format:
                normalized_format = {"geotiff": "tif", "tiff": "tif", "gpkg": "gpkg", "geojson": "geojson", "csv": "csv"}.get(requested_format, requested_format)
                if normalized_format in entry_format or file_name.endswith("." + normalized_format):
                    score += 200
                else:
                    score -= 80
            for year in requested_years:
                score += 180 if year in text else -40

            token_groups = {
                "weather": ("weather", "temperature", "precip", "tmax", "prcp", "monthly"),
                "acs": ("acs", "income", "rent", "population", "census attribute"),
                "tract": ("tract", "census tract"),
                "hospital": ("hospital",),
                "dem": ("dem", "elevation", "raster"),
                "boundary": ("boundary", "state college", "bellefonte"),
                "subdivision": ("subdivision", "municipality"),
            }
            for label, tokens in token_groups.items():
                if any(token in retrieval_text for token in tokens):
                    score += 80 if label in text or any(token in text for token in tokens) else -30

            if any(token in retrieval_text for token in ("tract", "census tract")):
                score += 400 if ("tract" in text or "census" in text) else -300
            if any(token in retrieval_text for token in ("raster", "elevation", "dem", "geotiff", "tif")):
                score += 500 if (entry_format in {"tif", "tiff", "geotiff"} or file_name.endswith((".tif", ".tiff"))) else -400
            # Retrieval tasks commonly describe the downstream contract in natural
            # language (for example, "CSV table input") rather than as an explicit
            # `format` parameter.  Treat that contract as a strong deterministic
            # ranking signal so a compatible tabular candidate wins over a generic
            # catalog default such as a DEM.
            if any(token in retrieval_text for token in ("csv", "table", "tabular")):
                score += 350 if (entry_format == "csv" or file_name.endswith(".csv")) else -150
            if any(token in retrieval_text for token in ("vector", "polygon", "boundary", "gpkg", "geopackage")):
                score += 120 if (entry_format == "gpkg" or file_name.endswith(".gpkg")) else 0
            if "state college" in retrieval_text or "state_college" in retrieval_text:
                score += 300 if ("state_college" in text or "state college" in text) else 0
                score -= 300 if "bellefonte" in text else 0
            if "bellefonte" in retrieval_text:
                score += 300 if "bellefonte" in text else 0
                score -= 300 if ("state_college" in text or "state college" in text) else 0

            if "30m" in retrieval_text or "30 m" in retrieval_text:
                score += 120 if "30m" in text else -25
            if "90m" in retrieval_text or "90 m" in retrieval_text:
                score += 120 if "90m" in text else -25
            if "utm" in retrieval_text or "26918" in retrieval_text:
                score += 40 if "utm" in text or "26918" in text else 0
            if "wgs" in retrieval_text or "4326" in retrieval_text:
                score += 40 if "wgs" in text or "4326" in text else 0

            return score

        # Match entry using a deterministic ranked retrieval policy instead of
        # returning the first non-excluded catalog row.  This keeps the worker
        # deterministic while letting the orchestrator's task parameters/query
        # control candidate selection.
        ranked_entries: list[tuple[int, int, dict[str, Any]]] = []
        for idx, entry in enumerate(catalog_entries):
            entry_id = str(entry.get("artifact_id") or entry.get("id") or "").lower()
            file_name = str(entry.get("file_name") or entry.get("filename") or "").lower()
            score = _score_entry(entry)
            if entry_id in exclude_ids or any(ex in file_name for ex in exclude_ids):
                score -= 1000
            ranked_entries.append((score, -idx, entry))
        selected_entry = max(ranked_entries, default=(0, 0, None))[2]

        if not selected_entry:
            return {
                "error": "E_DATASET_NOT_FOUND: No matching dataset found in the current catalog.",
                "summary": "No matching dataset available.",
                "artifacts": [],
            }

        src_path = Path(selected_entry.get("file_path", ""))
        if not src_path.is_absolute():
            base_dir = Path(__file__).resolve().parents[2] / "benchmark" / "fixtures"
            src_path = base_dir / src_path

        if not src_path.exists():
            fallback = Path(__file__).resolve().parents[2] / "benchmark" / "fixtures" / src_path.name
            if fallback.exists():
                src_path = fallback

        if not src_path.exists():
            return {
                "error": f"E_DATASET_UNREADABLE: Catalog entry source file '{src_path.name}' does not exist.",
                "summary": "Source dataset could not be opened.",
                "artifacts": [],
            }

        dest_filename = f"retrieved_{uuid.uuid4().hex[:8]}_{src_path.name}"
        dest_path = output_dir / dest_filename
        shutil.copy2(src_path, dest_path)

        return {
            "summary": f"Successfully retrieved dataset: {src_path.name}",
            "artifacts": [str(dest_path)],
        }


# ==============================================================================
# A2: Vector Analysis A
# ==============================================================================

class ControlledVectorAnalysisAAgent(GeoAgent):
    """Deterministic vector geoprocessing: buffer, clip, overlay, geometry repair, spatial join."""

    agent_id: ClassVar[str] = "vector_analysis_a"
    agent_name: ClassVar[str] = "Vector Analysis A"
    agent_version: ClassVar[str] = "2.0.0"
    agent_description: ClassVar[str] = (
        "Performs deterministic vector geoprocessing including buffering, clipping, "
        "overlay/intersection, explicit geometry repair, and spatial joins."
    )
    requires_input_datasets: ClassVar[bool] = True
    requires_model_credentials: ClassVar[bool] = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(output_dir=DATA_DIR / self.agent_id, **kwargs)

    def run(
        self,
        query: str,
        input_dataset_paths: list[str] | str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = self.normalize_dataset_paths(input_dataset_paths)

        if not paths:
            return {"error": "E_INPUT_MISSING: At least one input vector dataset is required.", "artifacts": []}

        lower_query = query.lower()
        params = dict(self.request_parameters or {})
        operation = str(params.get("operation") or "").lower()

        try:
            gdf1 = gpd.read_file(paths[0])
        except Exception as e:
            return {"error": f"E_FORMAT_UNSUPPORTED: Unable to open input vector file: {e}", "artifacts": []}

        # 1. Geometry Repair
        if "repair" in lower_query or "make_valid" in lower_query or operation == "geometry_repair":
            gdf1["geometry"] = gdf1["geometry"].apply(lambda g: make_valid(g) if g is not None else None)
            out_file = output_dir / f"repaired_{uuid.uuid4().hex[:8]}.gpkg"
            gdf1.to_file(out_file, driver="GPKG")
            return {"summary": f"Repaired geometries for {len(gdf1)} features.", "artifacts": [str(out_file)]}

        # Validate geometry validity for remaining operations
        if not gdf1.is_valid.all():
            return {
                "error": "E_INVALID_GEOMETRY: Input vector contains invalid geometry.",
                "artifacts": [],
            }

        # 2. Buffer
        if "buffer" in lower_query or operation == "buffer":
            dist = _extract_distance_meters(query, params)
            if dist is None or dist <= 0:
                return {"error": "E_PARAM_INVALID: A positive buffer distance is required.", "artifacts": []}

            if gdf1.crs is None:
                return {
                    "error": "E_CRS_MISSING: Metric buffer input has no coordinate reference system.",
                    "artifacts": [],
                }
            if gdf1.crs.is_geographic:
                return {
                    "error": (
                        "E_CRS_MISMATCH: Buffer operation with metric distance requires a projected CRS. "
                        f"Input dataset has geographic CRS: {gdf1.crs}."
                    ),
                    "artifacts": [],
                }

            buffered_geom = gdf1.geometry.buffer(dist)
            res_gdf = gdf1.copy()
            res_gdf.geometry = buffered_geom
            out_file = output_dir / f"buffer_{int(dist)}m_{uuid.uuid4().hex[:8]}.gpkg"
            res_gdf.to_file(out_file, driver="GPKG")
            return {"summary": f"Created {dist}m buffer for {len(res_gdf)} features.", "artifacts": [str(out_file)]}

        # Two-dataset operations
        if len(paths) < 2:
            return {"error": "E_INPUT_MISSING: Two input vector datasets required for overlay/clip/join.", "artifacts": []}

        try:
            gdf2 = gpd.read_file(paths[1])
        except Exception as e:
            return {"error": f"E_FORMAT_UNSUPPORTED: Unable to open second vector dataset: {e}", "artifacts": []}

        if not gdf2.is_valid.all():
            return {"error": "E_INVALID_GEOMETRY: Second input vector contains invalid geometry.", "artifacts": []}

        # CRS compatibility check
        if gdf1.crs != gdf2.crs:
            return {
                "error": (
                    f"E_CRS_MISMATCH: Binary spatial operation received inputs with different CRS identifiers "
                    f"({gdf1.crs} vs {gdf2.crs})."
                ),
                "artifacts": [],
            }

        # 3. Clip
        if "clip" in lower_query or operation == "clip":
            clipped = gpd.clip(gdf1, gdf2)
            out_file = output_dir / f"clipped_{uuid.uuid4().hex[:8]}.gpkg"
            clipped.to_file(out_file, driver="GPKG")
            return {"summary": f"Clipped vector dataset resulting in {len(clipped)} features.", "artifacts": [str(out_file)]}

        # 4. Overlay / Intersection
        if "intersect" in lower_query or "overlay" in lower_query or operation in {"overlay", "intersection"}:
            inter = gpd.overlay(gdf1, gdf2, how="intersection")
            out_file = output_dir / f"intersect_{uuid.uuid4().hex[:8]}.gpkg"
            inter.to_file(out_file, driver="GPKG")
            return {"summary": f"Intersected vectors resulting in {len(inter)} features.", "artifacts": [str(out_file)]}

        # 5. Spatial Join
        if "join" in lower_query or "count" in lower_query or "within" in lower_query or operation == "spatial_join":
            joined = gpd.sjoin(gdf1, gdf2, how="inner", predicate="intersects")
            out_file = output_dir / f"sjoin_{uuid.uuid4().hex[:8]}.gpkg"
            joined.to_file(out_file, driver="GPKG")
            return {"summary": f"Spatial join produced {len(joined)} matched features.", "artifacts": [str(out_file)]}

        return {"error": "E_OPERATION_UNRECOGNIZED: No supported vector operation detected in query.", "artifacts": []}


# ==============================================================================
# A3: Vector Analysis B
# ==============================================================================

class ControlledVectorAnalysisBAgent(GeoAgent):
    """Deterministic vector joins, dissolves, grouped aggregations, attribute joins, and buffering."""

    agent_id: ClassVar[str] = "vector_analysis_b"
    agent_name: ClassVar[str] = "Vector Analysis B"
    agent_version: ClassVar[str] = "2.0.0"
    agent_description: ClassVar[str] = (
        "Performs deterministic vector joins, dissolves, grouped aggregations, attribute joins, "
        "and buffering. Consumes GeoPackage inputs for vector workflows."
    )
    requires_input_datasets: ClassVar[bool] = True
    requires_model_credentials: ClassVar[bool] = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(output_dir=DATA_DIR / self.agent_id, **kwargs)

    def run(
        self,
        query: str,
        input_dataset_paths: list[str] | str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = self.normalize_dataset_paths(input_dataset_paths)

        if not paths:
            return {"error": "E_INPUT_MISSING: Input dataset required.", "artifacts": []}

        # Vector B strictly accepts GPKG (or CSV for pure tabular join inputs)
        p1 = Path(paths[0])
        if p1.suffix.lower() not in {".gpkg", ".csv"}:
            return {
                "error": f"E_FORMAT_UNSUPPORTED: Vector Analysis B accepts GeoPackage (.gpkg) inputs, received {p1.suffix}.",
                "artifacts": [],
            }

        params = dict(self.request_parameters or {})
        operation = str(params.get("operation") or "").lower()
        lower_query = query.lower()

        # 1. Attribute Join (Vector + Table)
        if "attribute_join" in operation or ("join" in lower_query and len(paths) >= 2 and any(p.endswith(".csv") for p in paths)):
            vector_path = next((p for p in paths if p.endswith(".gpkg")), None)
            table_path = next((p for p in paths if p.endswith(".csv")), None)
            if not vector_path or not table_path:
                return {"error": "E_INPUT_MISSING: Attribute join requires one GPKG vector and one CSV table.", "artifacts": []}

            gdf = gpd.read_file(vector_path)
            df = pd.read_csv(table_path)

            v_key = params.get("vector_key")
            t_key = params.get("table_key")

            if not v_key or not t_key:
                # Deduce keys from column overlap or common IDs
                for candidate in ["GEOID", "geoid", "tract_id", "id", "COUNTYFP", "FIPS"]:
                    if candidate in gdf.columns and candidate in df.columns:
                        v_key, t_key = candidate, candidate
                        break
                    elif candidate in gdf.columns:
                        match = next((c for c in df.columns if c.lower() in candidate.lower() or candidate.lower() in c.lower()), None)
                        if match:
                            v_key, t_key = candidate, match
                            break

            if not v_key or not t_key or v_key not in gdf.columns or t_key not in df.columns:
                return {
                    "error": f"E_SCHEMA_MISMATCH: Could not resolve join keys between vector columns {list(gdf.columns)} and table columns {list(df.columns)}.",
                    "artifacts": [],
                }

            gdf[v_key] = gdf[v_key].astype(str).str.strip()
            df[t_key] = df[t_key].astype(str).str.strip()

            joined = gdf.merge(df, left_on=v_key, right_on=t_key, how="inner")
            out_file = output_dir / f"attr_joined_{uuid.uuid4().hex[:8]}.gpkg"
            joined.to_file(out_file, driver="GPKG")
            return {"summary": f"Joined attributes by {v_key}={t_key}, producing {len(joined)} features.", "artifacts": [str(out_file)]}

        # Load primary vector dataset
        gdf1 = gpd.read_file(paths[0])

        # 2. Dissolve
        if "dissolve" in lower_query or operation == "dissolve":
            by_col = params.get("by")
            if by_col and by_col in gdf1.columns:
                dissolved = gdf1.dissolve(by=by_col, as_index=False)
            else:
                dissolved = gdf1.dissolve(as_index=False)
            out_file = output_dir / f"dissolved_{uuid.uuid4().hex[:8]}.gpkg"
            dissolved.to_file(out_file, driver="GPKG")
            return {"summary": f"Dissolved into {len(dissolved)} feature(s).", "artifacts": [str(out_file)]}

        # 3. Buffer
        if "buffer" in lower_query or operation == "buffer":
            dist = _extract_distance_meters(query, params)
            if dist is None or dist <= 0:
                return {"error": "E_PARAM_INVALID: A positive buffer distance is required.", "artifacts": []}

            if gdf1.crs is not None and gdf1.crs.is_geographic:
                return {
                    "error": (
                        "E_CRS_MISMATCH: Buffer operation with metric distance requires a projected CRS. "
                        f"Input dataset has geographic CRS: {gdf1.crs}."
                    ),
                    "artifacts": [],
                }

            res_gdf = gdf1.copy()
            res_gdf.geometry = gdf1.geometry.buffer(dist)
            out_file = output_dir / f"buffer_b_{int(dist)}m_{uuid.uuid4().hex[:8]}.gpkg"
            res_gdf.to_file(out_file, driver="GPKG")
            return {"summary": f"Created {dist}m buffer for {len(res_gdf)} features.", "artifacts": [str(out_file)]}

        # 4. Single-layer polygon attribute/area aggregation.  This covers
        # requests such as "population and land area by census tract" where the
        # input is already one feature per aggregation unit after any attribute
        # join.  It preserves feature granularity and adds deterministic area
        # fields rather than requiring a second spatial input.
        if operation in {"aggregate", "summarize", "summary"} and len(paths) == 1:
            if not all(str(t).lower() in {"polygon", "multipolygon"} for t in gdf1.geometry.geom_type.dropna().unique()):
                return {
                    "error": "E_INPUT_TYPE_MISMATCH: Single-layer aggregate requires a polygon layer.",
                    "artifacts": [],
                }
            out_gdf = gdf1.copy()
            if out_gdf.crs is not None and out_gdf.crs.is_geographic:
                metric_gdf = out_gdf.to_crs("EPSG:3857")
                area = metric_gdf.geometry.area
            else:
                area = out_gdf.geometry.area
            area_field = str(params.get("area_field") or "land_area_sq_m")
            out_gdf[area_field] = area.astype(float)
            out_gdf["land_area_km2"] = out_gdf[area_field] / 1_000_000.0

            population_field = (
                params.get("population_field")
                or next((c for c in out_gdf.columns if str(c).lower() in {"total_pop", "population", "pop"}), None)
            )
            if population_field and population_field in out_gdf.columns:
                out_gdf[population_field] = pd.to_numeric(out_gdf[population_field], errors="coerce")

            out_file = output_dir / f"polygon_aggregate_b_{uuid.uuid4().hex[:8]}.gpkg"
            out_gdf.to_file(out_file, driver="GPKG")
            return {
                "summary": (
                    f"Computed polygon-level aggregate fields for {len(out_gdf)} features"
                    + (f", preserving population field {population_field}." if population_field else ".")
                ),
                "artifacts": [str(out_file)],
            }

        # 5. Point-by-polygon aggregation
        if len(paths) >= 2 and operation in {
            "point_count_by_polygon",
            "point_sum_by_polygon",
            "aggregate",
        }:
            p2 = Path(paths[1])
            if p2.suffix.lower() != ".gpkg":
                return {"error": f"E_FORMAT_UNSUPPORTED: Vector B requires GeoPackage (.gpkg), received {p2.suffix}.", "artifacts": []}

            gdf2 = gpd.read_file(paths[1])
            if gdf1.crs != gdf2.crs:
                return {
                    "error": f"E_CRS_MISMATCH: Input layers have different coordinate references ({gdf1.crs} vs {gdf2.crs}).",
                    "artifacts": [],
                }

            inputs = [gdf1, gdf2]
            polygon_gdf = next(
                (
                    gdf
                    for gdf in inputs
                    if all(str(t).lower() in {"polygon", "multipolygon"} for t in gdf.geometry.geom_type.dropna().unique())
                ),
                None,
            )
            point_gdf = next(
                (
                    gdf
                    for gdf in inputs
                    if all(str(t).lower() in {"point", "multipoint"} for t in gdf.geometry.geom_type.dropna().unique())
                ),
                None,
            )
            if polygon_gdf is None or point_gdf is None:
                return {
                    "error": "E_INPUT_TYPE_MISMATCH: Point-count aggregation requires one polygon layer and one point layer.",
                    "artifacts": [],
                }

            count_field = str(params.get("count_field") or "hospital_count")
            sum_field = params.get("sum_field")
            output_field = str(params.get("output_field") or (f"{sum_field}_sum" if sum_field else count_field))
            predicate = str(params.get("predicate") or "intersects")

            polygons = polygon_gdf.copy().reset_index(drop=True)
            polygons["_gas_polygon_id"] = polygons.index
            joined = gpd.sjoin(
                point_gdf,
                polygons[["_gas_polygon_id", "geometry"]],
                how="inner",
                predicate=predicate,
            )
            if sum_field and sum_field in joined.columns:
                aggregates = joined.groupby("_gas_polygon_id")[sum_field].sum()
            else:
                aggregates = joined.groupby("_gas_polygon_id").size()

            polygons[output_field] = polygons["_gas_polygon_id"].map(aggregates).fillna(0)
            if not sum_field:
                polygons[output_field] = polygons[output_field].astype(int)
            polygons = polygons.drop(columns=["_gas_polygon_id"])
            out_file = output_dir / f"point_aggregate_b_{uuid.uuid4().hex[:8]}.gpkg"
            polygons.to_file(out_file, driver="GPKG")
            return {
                "summary": (
                    f"Aggregated {len(point_gdf)} point features into {len(polygons)} "
                    f"polygon units using field {output_field}."
                ),
                "artifacts": [str(out_file)],
            }

        if len(paths) >= 2 and ("join" in lower_query or "count" in lower_query or operation == "spatial_join"):
            p2 = Path(paths[1])
            if p2.suffix.lower() != ".gpkg":
                return {"error": f"E_FORMAT_UNSUPPORTED: Vector B requires GeoPackage (.gpkg), received {p2.suffix}.", "artifacts": []}

            gdf2 = gpd.read_file(paths[1])
            if gdf1.crs != gdf2.crs:
                return {
                    "error": f"E_CRS_MISMATCH: Input layers have different coordinate references ({gdf1.crs} vs {gdf2.crs}).",
                    "artifacts": [],
                }

            joined = gpd.sjoin(gdf1, gdf2, how="inner", predicate="intersects")
            out_file = output_dir / f"sjoin_b_{uuid.uuid4().hex[:8]}.gpkg"
            joined.to_file(out_file, driver="GPKG")
            return {"summary": f"Spatial join produced {len(joined)} matched features.", "artifacts": [str(out_file)]}

        return {"error": "E_OPERATION_UNRECOGNIZED: No supported Vector B operation requested.", "artifacts": []}


# ==============================================================================
# A4: Raster Analysis Agent
# ==============================================================================

class ControlledRasterAnalysisAgent(GeoAgent):
    """Deterministic raster analysis: reproject, resample, clip, raster calculator, zonal statistics."""

    agent_id: ClassVar[str] = "raster_analysis_agent"
    agent_name: ClassVar[str] = "Raster Analysis Agent"
    agent_version: ClassVar[str] = "2.0.0"
    agent_description: ClassVar[str] = (
        "Performs deterministic raster reprojection, resampling, clipping, raster calculation, "
        "and zonal statistics. Consumes GeoTIFF rasters and GeoPackage polygon zones."
    )
    requires_input_datasets: ClassVar[bool] = True
    requires_model_credentials: ClassVar[bool] = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(output_dir=DATA_DIR / self.agent_id, **kwargs)

    def run(
        self,
        query: str,
        input_dataset_paths: list[str] | str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = self.normalize_dataset_paths(input_dataset_paths)

        if not paths:
            return {"error": "E_INPUT_MISSING: Input dataset(s) required.", "artifacts": []}

        params = dict(self.request_parameters or {})
        operation = str(params.get("operation") or "").lower()
        lower_query = query.lower()

        # 1. Zonal Statistics (Raster + Polygon Zones)
        if "zonal" in lower_query or "summarize" in lower_query or "mean elevation" in lower_query or operation == "zonal_statistics":
            raster_path = next((p for p in paths if p.endswith(".tif") or p.endswith(".tiff")), None)
            zone_path = next((p for p in paths if p.endswith(".gpkg") or p.endswith(".geojson")), None)

            if not raster_path or not zone_path:
                return {"error": "E_INPUT_MISSING: Zonal statistics requires one raster (.tif) and one polygon zone layer.", "artifacts": []}

            # Enforce GeoPackage requirement for zones
            if not zone_path.endswith(".gpkg"):
                return {"error": f"E_FORMAT_UNSUPPORTED: Zonal statistics requires GeoPackage (.gpkg) zones, received {Path(zone_path).suffix}.", "artifacts": []}

            zones_gdf = gpd.read_file(zone_path)
            with rasterio.open(raster_path) as src:
                raster_crs = src.crs
                # Check CRS match
                if zones_gdf.crs != raster_crs:
                    return {
                        "error": f"E_CRS_MISMATCH: Raster CRS ({raster_crs}) does not match polygon zone CRS ({zones_gdf.crs}).",
                        "artifacts": [],
                    }

                results = []
                for idx, row in zones_gdf.iterrows():
                    geom = [mapping(row.geometry)]
                    try:
                        out_image, out_transform = rasterio.mask.mask(src, geom, crop=True, nodata=src.nodata or -9999)
                        valid_pixels = out_image[out_image != (src.nodata or -9999)]
                        mean_val = float(np.mean(valid_pixels)) if len(valid_pixels) > 0 else None
                        cell_cnt = int(len(valid_pixels))
                    except Exception:
                        mean_val = None
                        cell_cnt = 0

                    zone_id = row.get("GEOID") or row.get("NAME") or row.get("id") or str(idx)
                    results.append({"zone_id": zone_id, "cell_count": cell_cnt, "mean_elevation": mean_val})

            out_csv = output_dir / f"zonal_stats_{uuid.uuid4().hex[:8]}.csv"
            pd.DataFrame(results).to_csv(out_csv, index=False)
            return {"summary": f"Calculated zonal statistics for {len(results)} zones.", "artifacts": [str(out_csv)]}

        # 2. Raster Difference / Calculation (Two Rasters)
        if "difference" in lower_query or operation in {"raster_calculator", "difference"}:
            raster_paths = [p for p in paths if p.endswith(".tif") or p.endswith(".tiff")]
            if len(raster_paths) < 2:
                return {"error": "E_INPUT_MISSING: Raster difference calculation requires two input GeoTIFF rasters.", "artifacts": []}

            with rasterio.open(raster_paths[0]) as r1, rasterio.open(raster_paths[1]) as r2:
                if r1.crs != r2.crs:
                    return {"error": f"E_CRS_MISMATCH: Rasters have different coordinate systems ({r1.crs} vs {r2.crs}).", "artifacts": []}

                # Check grid alignment (dimensions and affine transform)
                if r1.shape != r2.shape or not np.allclose(r1.transform, r2.transform, atol=1e-3):
                    return {
                        "error": (
                            f"E_GRID_MISALIGNED: Rasters have incompatible grid geometries "
                            f"(Shape {r1.shape} vs {r2.shape}, Res {r1.res} vs {r2.res})."
                        ),
                        "artifacts": [],
                    }

                d1 = r1.read(1).astype(np.float32)
                d2 = r2.read(1).astype(np.float32)
                diff = np.abs(d1 - d2) if "absolute" in lower_query else (d1 - d2)

                out_meta = r1.meta.copy()
                out_meta.update(dtype=rasterio.float32, count=1)
                out_tif = output_dir / f"raster_diff_{uuid.uuid4().hex[:8]}.tif"
                with rasterio.open(out_tif, "w", **out_meta) as dst:
                    dst.write(diff, 1)

            return {"summary": "Computed cellwise raster difference.", "artifacts": [str(out_tif)]}

        # 3. Resample / Align Raster
        if "resample" in lower_query or "align" in lower_query or operation == "resample":
            match_grid = (
                params.get("match_grid")
                or params.get("target_grid")
                or params.get("reference_grid")
                or params.get("reference_input_role")
                or params.get("reference_grid_role")
            )
            if match_grid:
                r_path = paths[0]
                target_path = str(match_grid)
                if not Path(target_path).is_absolute():
                    matching = next(
                        (
                            p for p in paths[1:]
                            if Path(p).name == target_path
                            or Path(p).stem == target_path
                            or target_path.lower() in {"input_1", "reference", "reference_raster", "target"}
                        ),
                        None,
                    )
                    target_path = matching or target_path
                with rasterio.open(r_path) as src, rasterio.open(target_path) as target:
                    out_meta = src.meta.copy()
                    out_meta.update(
                        width=target.width,
                        height=target.height,
                        transform=target.transform,
                        crs=target.crs,
                    )
                    out_tif = output_dir / f"resampled_match_{uuid.uuid4().hex[:8]}.tif"
                    with rasterio.open(out_tif, "w", **out_meta) as dst:
                        for band_index in range(1, src.count + 1):
                            rasterio.warp.reproject(
                                source=rasterio.band(src, band_index),
                                destination=rasterio.band(dst, band_index),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=target.transform,
                                dst_crs=target.crs,
                                resampling=Resampling.bilinear,
                            )
                return {"summary": f"Resampled raster to match grid {Path(target_path).name}.", "artifacts": [str(out_tif)]}

            target_res = _first_number(params.get("resolution"), 10.0)
            r_path = paths[0]
            with rasterio.open(r_path) as src:
                scale_factor = src.res[0] / target_res
                new_width = int(src.width * scale_factor)
                new_height = int(src.height * scale_factor)
                new_transform = src.transform * src.transform.scale(
                    (src.width / new_width),
                    (src.height / new_height),
                )
                data = src.read(out_shape=(src.count, new_height, new_width), resampling=Resampling.bilinear)
                out_meta = src.meta.copy()
                out_meta.update(width=new_width, height=new_height, transform=new_transform)

                out_tif = output_dir / f"resampled_{int(target_res)}m_{uuid.uuid4().hex[:8]}.tif"
                with rasterio.open(out_tif, "w", **out_meta) as dst:
                    dst.write(data)

            return {"summary": f"Resampled raster to {target_res}m resolution.", "artifacts": [str(out_tif)]}

        # 4. Clip Raster
        if "clip" in lower_query or operation == "clip":
            r_path = next((p for p in paths if p.endswith(".tif") or p.endswith(".tiff")), None)
            v_path = next((p for p in paths if p.endswith(".gpkg") or p.endswith(".geojson")), None)
            if not r_path or not v_path:
                return {"error": "E_INPUT_MISSING: Raster clipping requires one GeoTIFF and one vector boundary layer.", "artifacts": []}

            poly_gdf = gpd.read_file(v_path)
            if not poly_gdf.is_valid.all():
                return {"error": "E_INVALID_GEOMETRY: Clipping boundary polygon geometry is invalid.", "artifacts": []}

            with rasterio.open(r_path) as src:
                if poly_gdf.crs != src.crs:
                    return {"error": f"E_CRS_MISMATCH: Boundary CRS ({poly_gdf.crs}) does not match raster CRS ({src.crs}).", "artifacts": []}

                geoms = [mapping(g) for g in poly_gdf.geometry if g is not None]
                out_image, out_transform = rasterio.mask.mask(src, geoms, crop=True)
                out_meta = src.meta.copy()
                out_meta.update(
                    driver="GTiff",
                    height=out_image.shape[1],
                    width=out_image.shape[2],
                    transform=out_transform,
                )
                out_tif = output_dir / f"clipped_{uuid.uuid4().hex[:8]}.tif"
                with rasterio.open(out_tif, "w", **out_meta) as dst:
                    dst.write(out_image)

            return {"summary": "Clipped raster to boundary extent.", "artifacts": [str(out_tif)]}

        # 5. Reproject Raster
        if "reproject" in lower_query or operation == "reproject_raster":
            target_crs = _extract_target_crs(query, params)
            if not target_crs:
                return {"error": "E_PARAM_MISSING: Target CRS is required for raster reprojection.", "artifacts": []}

            r_path = paths[0]
            with rasterio.open(r_path) as src:
                transform, width, height = rasterio.warp.calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )
                kwargs = src.meta.copy()
                kwargs.update(crs=target_crs, transform=transform, width=width, height=height)
                out_tif = output_dir / f"reprojected_{uuid.uuid4().hex[:8]}.tif"
                with rasterio.open(out_tif, "w", **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        rasterio.warp.reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.bilinear,
                        )
            return {"summary": f"Reprojected raster to {target_crs}.", "artifacts": [str(out_tif)]}

        return {"error": "E_OPERATION_UNRECOGNIZED: No valid raster operation identified in query.", "artifacts": []}


# ==============================================================================
# A5: Projection Agent
# ==============================================================================

class ControlledProjectionAgent(GeoAgent):
    """Vector reprojection and explicit CRS assignment."""

    agent_id: ClassVar[str] = "projection_agent"
    agent_name: ClassVar[str] = "Projection Agent"
    agent_version: ClassVar[str] = "2.0.0"
    agent_description: ClassVar[str] = (
        "Reprojects vector artifacts between explicitly identified coordinate reference systems "
        "and assigns a CRS when explicitly supplied. The service never guesses an unknown CRS."
    )
    requires_input_datasets: ClassVar[bool] = True
    requires_model_credentials: ClassVar[bool] = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(output_dir=DATA_DIR / self.agent_id, **kwargs)

    def run(
        self,
        query: str,
        input_dataset_paths: list[str] | str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = self.normalize_dataset_paths(input_dataset_paths)

        if not paths:
            return {"error": "E_INPUT_MISSING: Vector dataset path is required.", "artifacts": []}

        params = dict(self.request_parameters or {})
        operation = str(params.get("operation") or "").lower()
        lower_query = query.lower()

        try:
            gdf = gpd.read_file(paths[0])
        except Exception as e:
            return {"error": f"E_FORMAT_UNSUPPORTED: Unable to open vector layer: {e}", "artifacts": []}

        target_crs = _extract_target_crs(query, params)

        # 1. Assign CRS (Metadata only, coordinates unchanged)
        if "assign" in lower_query or operation == "assign_crs":
            if not target_crs:
                return {
                    "error": "E_MISSING_CRS: Explicit CRS parameter is required for assign_crs. Service does not guess unknown CRS.",
                    "artifacts": [],
                }
            gdf.set_crs(target_crs, allow_override=True, inplace=True)
            out_file = output_dir / f"assigned_crs_{uuid.uuid4().hex[:8]}.gpkg"
            gdf.to_file(out_file, driver="GPKG")
            return {"summary": f"Assigned CRS {target_crs} without transforming coordinates.", "artifacts": [str(out_file)]}

        # 2. Reproject (Coordinate transformation)
        if "reproject" in lower_query or "transform" in lower_query or operation == "reproject":
            if gdf.crs is None:
                return {
                    "error": "E_UNKNOWN_SOURCE_CRS: Input vector has undefined CRS (crs=None). Cannot reproject without known source CRS.",
                    "artifacts": [],
                }
            if not target_crs:
                return {"error": "E_PARAM_MISSING: Explicit target CRS is required for reprojection.", "artifacts": []}

            reprojected = gdf.to_crs(target_crs)
            out_file = output_dir / f"reprojected_{uuid.uuid4().hex[:8]}.gpkg"
            reprojected.to_file(out_file, driver="GPKG")
            return {"summary": f"Reprojected {len(reprojected)} features to {target_crs}.", "artifacts": [str(out_file)]}

        # Default fallback to reprojection if target CRS was found
        if target_crs and gdf.crs is not None:
            reprojected = gdf.to_crs(target_crs)
            out_file = output_dir / f"reprojected_{uuid.uuid4().hex[:8]}.gpkg"
            reprojected.to_file(out_file, driver="GPKG")
            return {"summary": f"Reprojected {len(reprojected)} features to {target_crs}.", "artifacts": [str(out_file)]}

        return {"error": "E_OPERATION_UNRECOGNIZED: Specify whether to reproject or assign CRS with target CRS.", "artifacts": []}


# ==============================================================================
# A6: Conversion Agent
# ==============================================================================

class ControlledConversionAgent(GeoAgent):
    """Converts artifact formats and constructs points from coordinate tables."""

    agent_id: ClassVar[str] = "conversion_agent"
    agent_name: ClassVar[str] = "Conversion Agent"
    agent_version: ClassVar[str] = "2.0.0"
    agent_description: ClassVar[str] = (
        "Converts supported artifact representations without changing analytical meaning. "
        "CSV-to-points requires coordinate column names and source CRS."
    )
    requires_input_datasets: ClassVar[bool] = True
    requires_model_credentials: ClassVar[bool] = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(output_dir=DATA_DIR / self.agent_id, **kwargs)

    def run(
        self,
        query: str,
        input_dataset_paths: list[str] | str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = self.normalize_dataset_paths(input_dataset_paths)

        if not paths:
            return {"error": "E_INPUT_MISSING: Input dataset required for conversion.", "artifacts": []}

        src_path = Path(paths[0])
        params = dict(self.request_parameters or {})
        operation = str(params.get("operation") or "").lower()
        lower_query = query.lower()

        # 1. CSV to Points
        if src_path.suffix.lower() == ".csv" or "csv_to_points" in operation or "point" in lower_query:
            try:
                df = pd.read_csv(src_path)
            except Exception as e:
                return {"error": f"E_FORMAT_UNSUPPORTED: Could not read CSV table: {e}", "artifacts": []}

            # Locate latitude and longitude columns
            lat_col = params.get("lat_col") or next((c for c in df.columns if c.upper() in {"LATITUDE", "LAT", "Y"}), None)
            lon_col = params.get("lon_col") or next((c for c in df.columns if c.upper() in {"LONGITUDE", "LON", "LNG", "X"}), None)

            if not lat_col or not lon_col:
                return {
                    "error": f"E_SCHEMA_MISMATCH: Missing latitude/longitude columns in CSV (found {list(df.columns)}).",
                    "artifacts": [],
                }

            crs = _extract_target_crs(query, params) or "EPSG:4326"
            geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)

            out_file = output_dir / f"spatialized_points_{uuid.uuid4().hex[:8]}.gpkg"
            gdf.to_file(out_file, driver="GPKG")
            return {"summary": f"Constructed {len(gdf)} point features from CSV coordinates with CRS {crs}.", "artifacts": [str(out_file)]}

        # 2. GeoJSON to GeoPackage
        if src_path.suffix.lower() in {".geojson", ".json"} or "geojson_to_gpkg" in operation or "gpkg" in lower_query:
            try:
                gdf = gpd.read_file(src_path)
                out_file = output_dir / f"converted_{src_path.stem}_{uuid.uuid4().hex[:8]}.gpkg"
                gdf.to_file(out_file, driver="GPKG")
                return {"summary": f"Converted GeoJSON to GeoPackage ({len(gdf)} features).", "artifacts": [str(out_file)]}
            except Exception as e:
                return {"error": f"E_CONVERSION_FAILED: Failed to convert GeoJSON to GPKG: {e}", "artifacts": []}

        # 3. Vector to CSV (Table export)
        if "table" in lower_query or "csv" in lower_query or operation == "table_to_csv":
            try:
                gdf = gpd.read_file(src_path)
                df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
                out_csv = output_dir / f"table_{src_path.stem}_{uuid.uuid4().hex[:8]}.csv"
                df.to_csv(out_csv, index=False)
                return {"summary": f"Exported attribute table to CSV ({len(df)} rows).", "artifacts": [str(out_csv)]}
            except Exception as e:
                return {"error": f"E_CONVERSION_FAILED: Failed to export table to CSV: {e}", "artifacts": []}

        return {"error": "E_OPERATION_UNRECOGNIZED: No recognized conversion operation.", "artifacts": []}


# ==============================================================================
# A7: Statistics Agent
# ==============================================================================

class ControlledStatisticsAgent(GeoAgent):
    """Deterministic tabular summaries and grouped aggregations."""

    agent_id: ClassVar[str] = "statistics_agent"
    agent_name: ClassVar[str] = "Statistics Agent"
    agent_version: ClassVar[str] = "2.0.0"
    agent_description: ClassVar[str] = (
        "Performs deterministic tabular summaries and grouped aggregation on attribute datasets."
    )
    requires_input_datasets: ClassVar[bool] = True
    requires_model_credentials: ClassVar[bool] = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(output_dir=DATA_DIR / self.agent_id, **kwargs)

    def run(
        self,
        query: str,
        input_dataset_paths: list[str] | str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = self.normalize_dataset_paths(input_dataset_paths)

        if not paths:
            return {"error": "E_INPUT_MISSING: Input tabular dataset required.", "artifacts": []}

        src_path = Path(paths[0])
        try:
            if src_path.suffix.lower() == ".csv":
                df = pd.read_csv(src_path)
            else:
                gdf = gpd.read_file(src_path)
                df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        except Exception as e:
            return {"error": f"E_FORMAT_UNSUPPORTED: Could not read input dataset: {e}", "artifacts": []}

        params = dict(self.request_parameters or {})
        lower_query = query.lower()

        # 1. Grouped Aggregate (e.g. monthly weather summaries)
        group_col = params.get("group_by")
        if not group_col:
            if "month" in lower_query and "month" in df.columns:
                group_col = "month"
            elif "year" in lower_query and "year" in df.columns:
                group_col = "year"
            else:
                for cand in ["month", "tract_id", "zone_id", "GEOID", "year"]:
                    if cand in df.columns:
                        group_col = cand
                        break

        val_col = params.get("value_field")
        if not val_col:
            if "tmax" in lower_query or "temperature" in lower_query:
                val_col = next((c for c in df.columns if "tmax" in c.lower() or "temp" in c.lower()), None)
            elif "prcp" in lower_query or "precipitation" in lower_query:
                val_col = next((c for c in df.columns if "prcp" in c.lower() or "precip" in c.lower()), None)

        if group_col and group_col in df.columns:
            if val_col and val_col in df.columns:
                stat_func = params.get("statistic", "mean").lower()
                agg_df = df.groupby(group_col)[val_col].agg([stat_func]).reset_index()
                agg_df.columns = [group_col, f"{val_col}_{stat_func}"]
            else:
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                agg_df = df.groupby(group_col)[num_cols].mean().reset_index()

            out_csv = output_dir / f"summary_by_{group_col}_{uuid.uuid4().hex[:8]}.csv"
            agg_df.to_csv(out_csv, index=False)
            return {"summary": f"Computed grouped aggregate by {group_col} ({len(agg_df)} groups).", "artifacts": [str(out_csv)]}

        # 2. Summary Statistics
        stats_df = df.describe().reset_index()
        out_csv = output_dir / f"summary_stats_{uuid.uuid4().hex[:8]}.csv"
        stats_df.to_csv(out_csv, index=False)
        return {"summary": "Generated descriptive summary statistics.", "artifacts": [str(out_csv)]}


# ==============================================================================
# A8: Mapping Agent
# ==============================================================================

class ControlledMappingAgent(GeoAgent):
    """Deterministic thematic maps: choropleth, vector map, raster map."""

    agent_id: ClassVar[str] = "mapping_agent"
    agent_name: ClassVar[str] = "Mapping Agent"
    agent_version: ClassVar[str] = "2.0.0"
    agent_description: ClassVar[str] = (
        "Produces deterministic HTML maps from supplied vector, raster, or joined tabular-spatial artifacts."
    )
    requires_input_datasets: ClassVar[bool] = True
    requires_model_credentials: ClassVar[bool] = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(output_dir=DATA_DIR / self.agent_id, **kwargs)

    def run(
        self,
        query: str,
        input_dataset_paths: list[str] | str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = self.normalize_dataset_paths(input_dataset_paths)

        if not paths:
            return {"error": "E_INPUT_MISSING: Input layer required for mapping.", "artifacts": []}

        src_path = Path(paths[0])
        params = dict(self.request_parameters or {})
        lower_query = query.lower()

        # HTML Map Generation
        out_html = output_dir / f"map_{uuid.uuid4().hex[:8]}.html"

        if src_path.suffix.lower() in {".gpkg", ".geojson"}:
            try:
                gdf = gpd.read_file(src_path)
            except Exception as e:
                return {"error": f"E_FORMAT_UNSUPPORTED: Could not open vector layer for mapping: {e}", "artifacts": []}

            # Reproject to WGS84 for Leaflet preview
            wgs_gdf = gdf.to_crs("EPSG:4326") if gdf.crs and not gdf.crs.is_geographic else gdf

            mapped_field = params.get("field")
            if not mapped_field:
                # Find numerical or attribute field
                candidates = [c for c in gdf.columns if c not in {"geometry", "FID", "id"}]
                if "income" in lower_query:
                    mapped_field = next((c for c in candidates if "income" in c.lower()), candidates[0] if candidates else None)
                elif "rent" in lower_query:
                    mapped_field = next((c for c in candidates if "rent" in c.lower()), candidates[0] if candidates else None)
                elif "hospital" in lower_query or "count" in lower_query:
                    mapped_field = next((c for c in candidates if "count" in c.lower() or "hospital" in c.lower()), candidates[0] if candidates else None)
                else:
                    mapped_field = candidates[0] if candidates else None

            bounds = wgs_gdf.total_bounds.tolist() if len(wgs_gdf) > 0 else [-77.9, 40.7, -77.7, 40.9]
            center_lat = (bounds[1] + bounds[3]) / 2.0
            center_lon = (bounds[0] + bounds[2]) / 2.0

            geojson_str = wgs_gdf.to_json()
            embedded_metadata = {
                "source_artifact": src_path.name,
                "mapped_field": mapped_field,
                "crs": str(gdf.crs),
                "feature_count": len(gdf),
                "bbox": bounds,
            }

            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>GAS Map Output - {src_path.name}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; }}
        #map {{ width: 100%; height: 100%; }}
        .meta-card {{ position: absolute; top: 10px; right: 10px; z-index: 1000; background: rgba(255,255,255,0.9); padding: 12px; border-radius: 6px; font-family: sans-serif; font-size: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.2); }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="meta-card">
        <strong>Layer:</strong> {src_path.name}<br/>
        <strong>Field:</strong> {mapped_field}<br/>
        <strong>Features:</strong> {len(gdf)}<br/>
        <strong>CRS:</strong> {gdf.crs}
    </div>
    <!-- EMBEDDED_METADATA: {json.dumps(embedded_metadata)} -->
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 11);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '© OpenStreetMap'
        }}).addTo(map);
        var geojsonData = {geojson_str};
        L.geoJSON(geojsonData, {{
            style: function(feature) {{
                return {{ color: "#2563eb", weight: 2, fillOpacity: 0.4 }};
            }}
        }}).addTo(map);
    </script>
</body>
</html>"""
            with open(out_html, "w", encoding="utf-8") as f:
                f.write(html_content)

            return {
                "summary": f"Rendered map with {len(gdf)} features (mapped field: {mapped_field}).",
                "artifacts": [str(out_html)],
            }

        return {"error": "E_FORMAT_UNSUPPORTED: Unsupported dataset format for mapping.", "artifacts": []}
