"""Bounded artifact-inspection policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class UnsafeArtifactPath(ValueError):
    pass


@dataclass(frozen=True)
class InspectionPolicy:
    allowed_roots: tuple[Path, ...] = ()
    max_file_bytes: int = 512 * 1024 * 1024
    max_schema_fields: int = 512
    max_features_to_validate: int = 100_000
    max_raster_bands: int = 256

    def validate_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_symlink():
            raise UnsafeArtifactPath("Symbolic-link artifacts are not inspected.")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise UnsafeArtifactPath("Artifact is not a regular file.")
        if resolved.stat().st_size > self.max_file_bytes:
            raise UnsafeArtifactPath(
                f"Artifact exceeds inspection limit of {self.max_file_bytes} bytes."
            )
        if self.allowed_roots:
            allowed = False
            for root in self.allowed_roots:
                try:
                    resolved.relative_to(root.resolve(strict=True))
                    allowed = True
                    break
                except ValueError:
                    continue
            if not allowed:
                raise UnsafeArtifactPath("Artifact is outside configured inspection roots.")
        return resolved
