from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProvenanceResult:
    p_fake: float | None = None
    evidence: list[str] = field(default_factory=list)
    override: str | None = None

    def as_dict(self) -> dict:
        return {
            "p_fake": self.p_fake,
            "evidence": self.evidence,
            "override": self.override,
        }


def inspect(_data: bytes) -> ProvenanceResult:
    return ProvenanceResult()
