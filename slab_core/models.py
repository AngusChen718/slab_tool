from dataclasses import dataclass, field
from typing import Any


@dataclass
class BuildCandidate:
    atoms: Any
    candidate_id: str
    builder: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    status: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)

    @property
    def passed(self):
        return self.status == "passed"

    def to_dict(self):
        return {
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "checks": self.checks,
        }


@dataclass
class AdsorbateSpec:
    name: str
    atoms: Any
    anchor_index: int
    charge: int = 0
    spin_multiplicity: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class AdsorptionSite:
    site_id: str
    site_type: str
    fractional_xy: tuple[float, float]
    cartesian: tuple[float, float, float]
    coordination: int
    surface_indices: list[int] = field(default_factory=list)
    multiplicity: int = 1
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "site_id": self.site_id,
            "site_type": self.site_type,
            "fractional_xy": list(self.fractional_xy),
            "cartesian": list(self.cartesian),
            "coordination": self.coordination,
            "surface_indices": self.surface_indices,
            "multiplicity": self.multiplicity,
            "metadata": self.metadata,
        }
