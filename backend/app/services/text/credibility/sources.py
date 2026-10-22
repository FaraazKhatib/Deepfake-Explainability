"""
Source credibility weighting.

This is the most ethically loaded component in the module, so its design is
deliberately constrained:

* **There is no blocklist.** The ratings file contains only positive entries.
  Nothing is labelled untrustworthy, because maintaining a list of "unreliable
  outlets" is a political act that a final-year project has no standing to
  perform, and getting it wrong silently suppresses legitimate reporting.

* **Ratings rest on structural, checkable properties**, not on perceived
  accuracy or political slant: is it a wire service, is it an IFCN Code of
  Principles signatory, is it a publicly funded broadcaster with a published
  complaints process, does it have a named masthead and a public corrections
  policy, is it peer-reviewed, is it the primary source for its own statements.

* **Unlisted means unrated, not untrustworthy.** An unrated domain gets
  ``UNRATED_SOURCE_WEIGHT`` (default 0.40) -- enough to contribute, not enough
  to outvote a wire service on its own. Most legitimate local and specialist
  outlets will be unrated; that is a limitation of the list, not a judgement
  about them, and the API says so.

* **The file is data, not code.** ``data/source_credibility.csv`` is meant to
  be edited, extended and argued with. Every row carries its rationale so a
  reader can check the reasoning rather than trusting the number.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .. import config
from ..retrieval.base import normalise_domain

logger = logging.getLogger(__name__)

UNRATED_TIER = "unrated"

# Domain suffixes that carry institutional accountability by construction.
# Applied only when the domain is not explicitly listed.
_SUFFIX_RULES = [
    (".gov", 0.85, "official_body", "Government domain; primary source for its own statements"),
    (".gov.uk", 0.85, "official_body", "Government domain; primary source for its own statements"),
    (".gov.in", 0.85, "official_body", "Government domain; primary source for its own statements"),
    (".gov.au", 0.85, "official_body", "Government domain; primary source for its own statements"),
    (".edu", 0.75, "academic", "Accredited academic institution domain"),
    (".ac.uk", 0.75, "academic", "Accredited academic institution domain"),
    (".ac.in", 0.75, "academic", "Accredited academic institution domain"),
    (".int", 0.80, "official_body", "Intergovernmental organisation domain"),
]


@dataclass
class SourceRating:
    """Credibility assessment for one domain."""

    domain: str
    tier: str
    weight: float
    rationale: str
    is_rated: bool

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "tier": self.tier,
            "weight": round(self.weight, 3),
            "rationale": self.rationale,
            "explicitly_rated": self.is_rated,
        }


class CredibilityIndex:
    """Loads the ratings file and answers lookups, with suffix fallbacks."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else self._default_path()
        self.ratings: Dict[str, SourceRating] = {}
        self.load_error: Optional[str] = None

    @staticmethod
    def _default_path() -> Path:
        # Resolve relative to the package root so the CWD does not matter.
        root = Path(__file__).resolve().parents[2]
        return root / config.CREDIBILITY_FILE

    def load(self) -> None:
        """Read the CSV. A missing file is survivable: everything becomes unrated."""
        if not self.path.exists():
            self.load_error = f"Credibility file not found at {self.path}"
            logger.warning(
                "%s -- every source will be treated as unrated (weight %.2f).",
                self.load_error, config.UNRATED_SOURCE_WEIGHT,
            )
            return

        try:
            with self.path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    domain = normalise_domain(row.get("domain", ""))
                    if not domain:
                        continue
                    try:
                        weight = float(row.get("weight", 0.5))
                    except ValueError:
                        weight = 0.5
                    self.ratings[domain] = SourceRating(
                        domain=domain,
                        tier=(row.get("tier") or UNRATED_TIER).strip(),
                        weight=max(0.0, min(1.0, weight)),
                        rationale=(row.get("rationale") or "").strip(),
                        is_rated=True,
                    )
            logger.info("Loaded %d source ratings from %s", len(self.ratings), self.path)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Could not read credibility file: %s", exc)

    def lookup(self, domain_or_url: str) -> SourceRating:
        """
        Rate a domain.

        Matching walks up the domain labels, so ``news.bbc.co.uk`` inherits the
        rating for ``bbc.co.uk`` without needing its own row.
        """
        domain = normalise_domain(domain_or_url)
        if not domain:
            return self._unrated("")

        if domain in self.ratings:
            return self.ratings[domain]

        labels = domain.split(".")
        for start in range(1, len(labels) - 1):
            parent = ".".join(labels[start:])
            if parent in self.ratings:
                rating = self.ratings[parent]
                return SourceRating(
                    domain=domain,
                    tier=rating.tier,
                    weight=rating.weight,
                    rationale=f"{rating.rationale} (inherited from {parent})",
                    is_rated=True,
                )

        for suffix, weight, tier, rationale in _SUFFIX_RULES:
            if domain.endswith(suffix):
                return SourceRating(
                    domain=domain, tier=tier, weight=weight,
                    rationale=rationale, is_rated=True,
                )

        return self._unrated(domain)

    @staticmethod
    def _unrated(domain: str) -> SourceRating:
        return SourceRating(
            domain=domain,
            tier=UNRATED_TIER,
            weight=config.UNRATED_SOURCE_WEIGHT,
            rationale=(
                "Not present in the ratings file. This is not a judgement that "
                "the source is unreliable -- most legitimate local and "
                "specialist outlets are simply unlisted."
            ),
            is_rated=False,
        )

    def info(self) -> dict:
        tiers: Dict[str, int] = {}
        for rating in self.ratings.values():
            tiers[rating.tier] = tiers.get(rating.tier, 0) + 1
        return {
            "file": str(self.path),
            "rated_domains": len(self.ratings),
            "tiers": tiers,
            "unrated_weight": config.UNRATED_SOURCE_WEIGHT,
            "load_error": self.load_error,
            "policy": (
                "Positive ratings only; no blocklist. Unlisted domains are "
                "unrated, not untrusted."
            ),
        }
