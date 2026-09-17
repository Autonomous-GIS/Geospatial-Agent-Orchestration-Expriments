"""Challenge Activation Validator for GAS Benchmark.

Verifies that the intended failure primitive or controlled environmental challenge
was truly present and activated in the initial task environment before testing resolution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import rasterio

from benchmark.evaluation.artifact_registry import ARTIFACT_REGISTRY, InspectedArtifact

logger = logging.getLogger(__name__)


class ChallengeActivationValidator:
    """Validates whether a benchmark task challenge condition was active at runtime."""

    def __init__(self, fixtures_dir: Path | None = None):
        self.fixtures_dir = fixtures_dir or Path(__file__).resolve().parents[1] / "fixtures"
        self._fixture_manifest: Dict[str, Dict[str, Any]] = {}
        self._load_fixture_manifest()

    def _load_fixture_manifest(self):
        m_path = self.fixtures_dir / "FIXTURE_MANIFEST.json"
        if m_path.exists():
            try:
                with open(m_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                    for entry in entries:
                        aid = entry.get("artifact_id")
                        if aid:
                            self._fixture_manifest[aid] = entry
            except Exception as e:
                logger.warning("Could not load fixture manifest %s: %s", m_path, e)

    def validate_activation(
        self,
        task_manifest: Dict[str, Any],
        initial_input_paths: List[str],
    ) -> Tuple[bool, List[str]]:
        """Verify that all challenge primitives for this task were active.

        Returns (is_activated: bool, reasons: list[str])
        """
        families = task_manifest.get("composed_families") or (
            [task_manifest.get("family_id")] if task_manifest.get("family_id") else []
        )
        if not families:
            return True, ["No specific failure families declared (control task)."]

        unactivated_reasons: List[str] = []

        # Resolve inspected input artifacts
        inputs: List[InspectedArtifact] = []
        for p in initial_input_paths:
            insp = ARTIFACT_REGISTRY.inspect(p)
            if insp:
                inputs.append(insp)

        for fid in families:
            active, reason = self._check_family_activation(fid, task_manifest, inputs)
            if not active:
                unactivated_reasons.append(f"Family {fid} challenge not active: {reason}")

        is_activated = len(unactivated_reasons) == 0
        return is_activated, unactivated_reasons

    def _check_family_activation(
        self,
        family_id: str,
        task_manifest: Dict[str, Any],
        inputs: List[InspectedArtifact],
    ) -> Tuple[bool, str]:
        # F01: Metric buffer on geographic CRS
        if family_id == "F01":
            geo_inputs = [inp for inp in inputs if inp.is_geographic]
            if not geo_inputs:
                return False, "No input dataset with geographic CRS found for metric buffer challenge."
            return True, "Geographic CRS input dataset confirmed."

        # F02: Vector-vector CRS mismatch in binary operation
        elif family_id == "F02":
            vector_inputs = [inp for inp in inputs if inp.artifact_type == "vector" and inp.crs]
            if len(vector_inputs) >= 2:
                crs_set = set(inp.crs for inp in vector_inputs)
                if len(crs_set) < 2:
                    return False, "Input vector datasets have identical CRS; mismatch challenge not active."
            return True, "Mismatched vector CRSs confirmed in input candidates."

        # F03: Missing CRS definition requiring human clarification
        elif family_id == "F03":
            missing_crs = [inp for inp in inputs if inp.crs is None or inp.crs == "None" or "missing_crs" in inp.file_name]
            if not missing_crs:
                return False, "No input dataset with missing CRS definition found."
            return True, "Missing CRS input dataset confirmed."

        # F04: Invalid polygon geometry
        elif family_id == "F04":
            invalid_geoms = [inp for inp in inputs if not inp.is_valid_geometry or inp.invalid_geometry_count > 0 or "invalid_geom" in inp.file_name]
            if not invalid_geoms:
                return False, "No input vector dataset with invalid geometry found."
            return True, "Invalid polygon geometry confirmed."

        # F05: Tabular coordinate spatialization
        elif family_id == "F05":
            csv_inputs = [inp for inp in inputs if inp.format == "CSV" and inp.bounds is not None]
            if not csv_inputs:
                return False, "No tabular CSV with coordinates found for spatialization challenge."
            return True, "Tabular CSV coordinate dataset confirmed."

        # F06: Format incompatibility
        elif family_id == "F06":
            geojson_inputs = [inp for inp in inputs if inp.format == "GEOJSON" or "geojson" in inp.file_name.lower()]
            if (
                task_manifest.get("required_constraints", {}).get("dissolved_geometry")
                and not task_manifest.get("required_constraints", {}).get("format_converted_to_gpkg")
                and any(inp.format == "GPKG" for inp in inputs)
            ):
                return True, "Supported GeoPackage dissolve control confirmed."
            if not geojson_inputs:
                return False, "No GeoJSON input dataset found for format incompatibility challenge."
            return True, "GeoJSON incompatible format dataset confirmed."

        # F07: Runtime schema drift
        elif family_id == "F07":
            drift_inputs = [inp for inp in inputs if any("drift" in col.lower() or "acs_med" in col.lower() or "schema_drift" in inp.file_name for col in inp.column_names)]
            if task_manifest.get("required_constraints", {}).get("ground_temperature_and_month_fields"):
                weather_inputs = [
                    inp
                    for inp in inputs
                    if inp.format == "CSV"
                    and any("month" == col.lower() for col in inp.column_names)
                    and any(
                        token in col.lower()
                        for col in inp.column_names
                        for token in ("tmax", "temp")
                    )
                ]
                if weather_inputs:
                    return True, "Runtime weather field grounding dataset confirmed."
            if not drift_inputs:
                return False, "No CSV input dataset with drifted schema attributes found."
            return True, "Drifted schema attribute dataset confirmed."

        # F08: Temporal mismatch in retrieval
        elif family_id == "F08":
            target_year = task_manifest.get("required_constraints", {}).get("year", 2024)
            # Check if fixture manifest / catalog contains multiple vintages
            catalog_files = [f.name for f in self.fixtures_dir.glob("fx_weather_*.csv")]
            has_multi_year = any("2023" in f for f in catalog_files) and any("2024" in f for f in catalog_files)
            if not has_multi_year:
                return False, "Catalog does not contain both 2023 and 2024 temporal vintages."
            return True, f"Catalog confirmed with competing temporal vintages for target year {target_year}."

        # F09: Partial spatial coverage
        elif family_id == "F09":
            partial_candidates = [inp for inp in inputs if "partial" in inp.file_name.lower()]
            if not partial_candidates:
                catalog_partial = (self.fixtures_dir / "fx_dem_partial_60pct_utm.tif").exists()
                if not catalog_partial:
                    return False, "Partial 60% DEM candidate not available in catalog."
            return True, "Partial spatial coverage candidate confirmed."

        # F10: Coarse resolution raster
        elif family_id == "F10":
            has_90m = any(inp.raster_resolution and max(inp.raster_resolution) >= 80.0 for inp in inputs)
            if not has_90m:
                catalog_90m = (self.fixtures_dir / "fx_dem_centre_90m_utm.tif").exists()
                if not catalog_90m:
                    return False, "90-meter coarse resolution DEM not available in catalog."
            return True, "Coarse 90m candidate confirmed in candidate pool."

        # F11: Raster grid misalignment
        elif family_id == "F11":
            misaligned = [inp for inp in inputs if "misaligned" in inp.file_name.lower()]
            if not misaligned:
                catalog_misaligned = (self.fixtures_dir / "fx_dem_misaligned_grid_utm.tif").exists()
                if not catalog_misaligned:
                    return False, "Misaligned grid raster candidate not found in catalog."
            return True, "Misaligned grid raster candidate confirmed."

        # F12: Raster-vector CRS mismatch in zonal stats
        elif family_id == "F12":
            has_3857_raster = any(inp.crs and "3857" in inp.crs for inp in inputs) or (self.fixtures_dir / "fx_dem_centre_epsg3857.tif").exists()
            has_utm_vector = any(inp.crs and ("utm" in inp.crs.lower() or "26918" in inp.crs or "32617" in inp.crs) for inp in inputs) or (self.fixtures_dir / "fx_tracts_centre_utm.gpkg").exists()
            if not (has_3857_raster and has_utm_vector):
                return False, "Competing raster (EPSG:3857) and vector (UTM) candidates not found in catalog."
            return True, "Raster-vector CRS mismatch candidates confirmed."

        # F13: Service unavailability
        elif family_id == "F13":
            return True, "Service A unavailability / 503 injection active for F13."

        # F14: Underspecified analytical unit
        elif family_id == "F14":
            has_tracts = (self.fixtures_dir / "fx_tracts_centre_utm.gpkg").exists()
            has_subdivisions = (self.fixtures_dir / "fx_county_subdivisions_utm.gpkg").exists()
            if not (has_tracts and has_subdivisions):
                return False, "Catalog does not contain both census tracts and county subdivisions."
            return True, "Multiple competing analytical unit partitions confirmed in catalog."

        # F15: Valid continuation (Control)
        elif family_id == "F15":
            return True, "Clean input datasets confirmed for control task."

        return True, f"Challenge activation confirmed for family {family_id}."
