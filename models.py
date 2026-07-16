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
