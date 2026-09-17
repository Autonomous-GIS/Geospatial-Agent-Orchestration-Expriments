"""Physical Artifact Registry and Deep Inspector for GAS Benchmark.

Inspects physical dataset artifacts produced during benchmark execution using GeoPandas,
Rasterio, Shapely, PyProj, and Pandas to establish ground-truth computational state.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class InspectedArtifact:
    """Detailed spatial, topological, raster, tabular, and temporal metadata."""

    path: str
    file_name: str
    sha256: str
    size_bytes: int
    format: str  # 'GPKG', 'GEOJSON', 'TIF', 'CSV', 'HTML', 'UNKNOWN'
    artifact_type: str  # 'vector', 'raster', 'table', 'map', 'other'
    crs: Optional[str] = None
    epsg_code: Optional[int] = None
    is_projected: Optional[bool] = None
    is_geographic: Optional[bool] = None
    bounds: Optional[List[float]] = None  # [minx, miny, maxx, maxy]
    geometry_types: List[str] = field(default_factory=list)
    feature_count: int = 0
    row_count: int = 0
    invalid_geometry_count: int = 0
    is_valid_geometry: bool = True
    schema: Dict[str, str] = field(default_factory=dict)
    column_names: List[str] = field(default_factory=list)
    raster_resolution: Optional[List[float]] = None  # [dx, dy]
    raster_dimensions: Optional[List[int]] = None  # [height, width, bands]
    nodata_value: Optional[float] = None
    cell_min: Optional[float] = None
    cell_max: Optional[float] = None
    cell_mean: Optional[float] = None
    temporal_years: List[int] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArtifactRegistry:
    """Registry and cache of inspected physical artifacts."""

    def __init__(self):
        self._cache: Dict[str, InspectedArtifact] = {}

    def inspect(self, file_path: Path | str | None) -> Optional[InspectedArtifact]:
        """Inspect a file and return its InspectedArtifact record."""
        if not file_path:
            return None
        p = Path(file_path).resolve()
        if not p.is_file():
            return None

        cache_key = f"{p}:{p.stat().st_mtime}:{p.stat().st_size}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        inspected = self._deep_inspect(p)
        self._cache[cache_key] = inspected
        return inspected

    def _deep_inspect(self, path: Path) -> InspectedArtifact:
        ext = path.suffix.lower()
        sha = compute_sha256(path)
        size = path.stat().st_size

        # Default fallback
        base = InspectedArtifact(
            path=str(path),
            file_name=path.name,
            sha256=sha,
            size_bytes=size,
            format="UNKNOWN",
            artifact_type="other",
        )

        try:
            if ext in {".gpkg", ".shp"}:
                return self._inspect_vector_geopandas(path, base, fmt="GPKG" if ext == ".gpkg" else "SHP")
            elif ext in {".geojson", ".json"}:
                return self._inspect_geojson(path, base)
            elif ext in {".tif", ".tiff"}:
                return self._inspect_raster(path, base)
            elif ext == ".csv":
                return self._inspect_csv(path, base)
            elif ext in {".html", ".htm"}:
                base.format = "HTML"
                base.artifact_type = "map"
                return base
            else:
                return base
        except Exception as exc:
            logger.warning("Error inspecting artifact %s: %s", path, exc)
            base.error = str(exc)
            return base

    def _inspect_vector_geopandas(self, path: Path, base: InspectedArtifact, fmt: str) -> InspectedArtifact:
        import geopandas as gpd
        from pyproj import CRS

        gdf = gpd.read_file(path)
        base.format = fmt
        base.artifact_type = "vector"
        base.feature_count = int(len(gdf))
        base.row_count = int(len(gdf))
        base.column_names = [str(c) for c in gdf.columns]
        base.schema = {str(c): str(dtype) for c, dtype in gdf.dtypes.items()}

        if gdf.crs is not None:
            try:
                crs_obj = CRS.from_user_input(gdf.crs)
                base.crs = crs_obj.to_string()
                base.epsg_code = crs_obj.to_epsg()
                base.is_projected = crs_obj.is_projected
                base.is_geographic = crs_obj.is_geographic
            except Exception:
                base.crs = str(gdf.crs)
                base.is_projected = False
                base.is_geographic = True

        if not gdf.empty:
            try:
                base.bounds = [float(v) for v in gdf.total_bounds]
            except Exception:
                pass

        if "geometry" in gdf and not gdf.empty:
            geom_types = set(str(t) for t in gdf.geometry.geom_type.dropna().unique())
            base.geometry_types = sorted(list(geom_types))

            try:
                invalid_mask = ~gdf.geometry.is_valid
                base.invalid_geometry_count = int(invalid_mask.sum())
                base.is_valid_geometry = bool(base.invalid_geometry_count == 0)
            except Exception:
                base.is_valid_geometry = False

        return base

    def _inspect_geojson(self, path: Path, base: InspectedArtifact) -> InspectedArtifact:
        import geopandas as gpd
        from pyproj import CRS

        try:
            gdf = gpd.read_file(path)
            base.format = "GEOJSON"
            base.artifact_type = "vector"
            base.feature_count = int(len(gdf))
            base.row_count = int(len(gdf))
            base.column_names = [str(c) for c in gdf.columns]
            base.schema = {str(c): str(dtype) for c, dtype in gdf.dtypes.items()}

            if gdf.crs is not None:
                try:
                    crs_obj = CRS.from_user_input(gdf.crs)
                    base.crs = crs_obj.to_string()
                    base.epsg_code = crs_obj.to_epsg()
                    base.is_projected = crs_obj.is_projected
                    base.is_geographic = crs_obj.is_geographic
                except Exception:
                    base.crs = str(gdf.crs)
            else:
                base.crs = "EPSG:4326"
                base.epsg_code = 4326
                base.is_geographic = True
                base.is_projected = False

            if not gdf.empty:
                base.bounds = [float(v) for v in gdf.total_bounds]

            if "geometry" in gdf and not gdf.empty:
                base.geometry_types = sorted(list(set(str(t) for t in gdf.geometry.geom_type.dropna().unique())))
                base.invalid_geometry_count = int((~gdf.geometry.is_valid).sum())
                base.is_valid_geometry = bool(base.invalid_geometry_count == 0)

            return base
        except Exception:
            # Fallback JSON parse
            base.format = "GEOJSON"
            base.artifact_type = "vector"
            return base

    def _inspect_raster(self, path: Path, base: InspectedArtifact) -> InspectedArtifact:
        import rasterio
        from pyproj import CRS

        with rasterio.open(path) as src:
            base.format = "TIF"
            base.artifact_type = "raster"
            base.raster_dimensions = [int(src.height), int(src.width), int(src.count)]
            base.raster_resolution = [float(abs(src.res[0])), float(abs(src.res[1]))]
            base.bounds = [float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)]
            base.nodata_value = float(src.nodata) if src.nodata is not None else None

            if src.crs is not None:
                try:
                    crs_obj = CRS.from_user_input(src.crs)
                    base.crs = crs_obj.to_string()
                    base.epsg_code = crs_obj.to_epsg()
                    base.is_projected = crs_obj.is_projected
                    base.is_geographic = crs_obj.is_geographic
                except Exception:
                    base.crs = str(src.crs)

            # Read first band sample for cell statistics
            try:
                data = src.read(1, masked=True)
                if data.count() > 0:
                    base.cell_min = float(data.min())
                    base.cell_max = float(data.max())
                    base.cell_mean = float(data.mean())
            except Exception:
                pass

        return base

    def _inspect_csv(self, path: Path, base: InspectedArtifact) -> InspectedArtifact:
        df = pd.read_csv(path, nrows=5000)
        base.format = "CSV"
        base.artifact_type = "table"
        base.row_count = int(len(df))
        base.feature_count = int(len(df))
        base.column_names = [str(c) for c in df.columns]
        base.schema = {str(c): str(dtype) for c, dtype in df.dtypes.items()}

        # Check for coordinates
        lon_candidates = [c for c in df.columns if c.lower() in {"lon", "lng", "longitude", "x", "long"}]
        lat_candidates = [c for c in df.columns if c.lower() in {"lat", "latitude", "y"}]

        if lon_candidates and lat_candidates:
            lon_col = lon_candidates[0]
            lat_col = lat_candidates[0]
            try:
                lons = pd.to_numeric(df[lon_col], errors="coerce").dropna()
                lats = pd.to_numeric(df[lat_col], errors="coerce").dropna()
                if not lons.empty and not lats.empty:
                    base.bounds = [float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max())]
                    base.crs = "EPSG:4326"
                    base.epsg_code = 4326
                    base.is_geographic = True
                    base.is_projected = False
                    base.geometry_types = ["Point"]
            except Exception:
                pass

        # Check for temporal years
        years_found = set()
        for col in df.columns:
            if "year" in col.lower() or "date" in col.lower():
                try:
                    for val in df[col].dropna().unique():
                        str_val = str(val)
                        match = re.search(r"\b(19\d\d|20\d\d)\b", str_val)
                        if match:
                            years_found.add(int(match.group(1)))
                except Exception:
                    pass
        base.temporal_years = sorted(list(years_found))

        return base


# Singleton registry
ARTIFACT_REGISTRY = ArtifactRegistry()
