"""
OntologyMappingEvaluator — accuracy measurement against a human-verified benchmark.

Architecture
────────────
  OntologyValidator  (validator.py) — checks code existence via live APIs
  MatchType          (enum)         — EXACT / WRONG / HALLUCINATED
  EvaluationDetail   (dataclass)    — one row of comparison output
  OntologyBreakdown  (dataclass)    — accuracy stats for one slice (ontology or entity)
  EvaluationMetrics  (dataclass)    — aggregate stats with per-ontology + per-entity views
  EvaluationReport   (dataclass)    — full report: metrics + detail rows + path
  OntologyMappingEvaluator (class)  — orchestrates load → map → score → report

Benchmark CSV format (minimum required columns)
───────────────────────────────────────────────
  source_variable   — source term passed to map_term()
  source_label      — human-readable label (optional)
  target_code       — ground-truth CURIE, e.g. "HP:0012735"
                      Pipe-separated for multi-concept variables: "HP:0001 | HP:0002"
  target_term       — ground-truth label (optional, used in reports)
  entity_type       — entity domain, e.g. "Phenotype" (optional per-row override)

Match semantics
───────────────
  EXACT        — predicted code(s) == ground-truth code(s)
  WRONG        — predicted code(s) exist in the ontology but are the wrong choice
  HALLUCINATED — predicted code(s) do not exist in any ontology (LLM invented them)

Dependencies
────────────
  pandas  — hard dep (install the 'eval' extra: pip install 'llm-ontology-mapper[eval]')
  requests — optional, needed only for OntologyValidator; imported lazily in validator.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from .validator import OntologyValidator

logger = logging.getLogger(__name__)



# ─────────────────────────────────────────────────────────────────────────────
# Match taxonomy
# ─────────────────────────────────────────────────────────────────────────────


class MatchType(str, Enum):
    """Classification of how a prediction compares with the ground truth."""

    EXACT       = "exact"       # Predicted code(s) == ground-truth code(s)
    WRONG       = "wrong"       # Code(s) exist but are the wrong choice (error)
    HALLUCINATED = "hallucinated"  # Code(s) do not exist in the ontology
    UNMAPPED    = "unmapped"    # Mapper returned UNMAPPED / ERROR sentinel
    UNKNOWN     = "unknown"     # Could not determine (API unavailable)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvaluationDetail:
    """Per-row evaluation output — one instance per benchmark variable."""

    source_variable:     str
    source_label:        str | None
    entity_type:         str | None
    ground_truth_codes:  list[str]         # normalised list (may be multi-code)
    ground_truth_term:   str | None
    predicted_code:      str
    predicted_term:      str
    predicted_confidence: float
    match_type:          MatchType
    notes:               str | None = None

    @property
    def is_correct(self) -> bool:
        return self.match_type == MatchType.EXACT

    @property
    def is_hallucinated(self) -> bool:
        return self.match_type == MatchType.HALLUCINATED

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_variable":      self.source_variable,
            "source_label":         self.source_label,
            "entity_type":          self.entity_type,
            "ground_truth_codes":   " | ".join(self.ground_truth_codes),
            "ground_truth_term":    self.ground_truth_term,
            "predicted_code":       self.predicted_code,
            "predicted_term":       self.predicted_term,
            "predicted_confidence": self.predicted_confidence,
            "match_type":           self.match_type.value,
            "is_correct":           self.is_correct,
            "is_hallucinated":      self.is_hallucinated,
            "notes":                self.notes,
        }


@dataclass
class OntologyBreakdown:
    """Accuracy stats for a single slice (one ontology prefix or one entity type)."""

    total:       int = 0
    exact:       int = 0
    wrong:       int = 0
    hallucinated: int = 0
    unmapped:    int = 0

    @property
    def accuracy(self) -> float:
        return self.exact / self.total if self.total else 0.0

    @property
    def wrong_rate(self) -> float:
        return self.wrong / self.total if self.total else 0.0

    @property
    def hallucination_rate(self) -> float:
        return self.hallucinated / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total":             self.total,
            "exact":             self.exact,
            "wrong":             self.wrong,
            "hallucinated":      self.hallucinated,
            "unmapped":          self.unmapped,
            "accuracy":          self.accuracy,
            "wrong_rate":        self.wrong_rate,
            "hallucination_rate": self.hallucination_rate,
        }


@dataclass
class EvaluationMetrics:
    """
    Aggregate evaluation metrics over a full benchmark run.

    Match taxonomy:
      EXACT        — predicted code(s) match ground truth exactly.
      WRONG        — code(s) exist in the ontology but are not the right choice.
      HALLUCINATED — code(s) do not exist at all.
      UNMAPPED     — mapper returned UNMAPPED / ERROR sentinel.
      UNKNOWN      — could not classify (API unavailable).
    """

    total:        int   = 0
    exact:        int   = 0
    wrong:        int   = 0
    hallucinated: int   = 0
    unmapped:     int   = 0
    unknown:      int   = 0
    avg_confidence: float = 0.0

    per_ontology: dict[str, OntologyBreakdown] = field(default_factory=dict)
    per_entity:   dict[str, OntologyBreakdown] = field(default_factory=dict)

    # ── Derived rates ─────────────────────────────────────────────────────────

    @property
    def accuracy(self) -> float:
        """Fraction of rows with an exact match."""
        return self.exact / self.total if self.total else 0.0

    @property
    def wrong_rate(self) -> float:
        """Fraction of rows where the code exists but is the wrong choice."""
        return self.wrong / self.total if self.total else 0.0

    @property
    def hallucination_rate(self) -> float:
        """Fraction of rows where the predicted code does not exist."""
        return self.hallucinated / self.total if self.total else 0.0

    @property
    def ontology_accuracy(self) -> float:
        """Fraction of rows that are either exact or wrong (i.e. a real code was produced)."""
        return (self.exact + self.wrong) / self.total if self.total else 0.0

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "total":             self.total,
            "exact":             self.exact,
            "wrong":             self.wrong,
            "hallucinated":      self.hallucinated,
            "unmapped":          self.unmapped,
            "unknown":           self.unknown,
            "accuracy":          self.accuracy,
            "wrong_rate":        self.wrong_rate,
            "hallucination_rate": self.hallucination_rate,
            "ontology_accuracy": self.ontology_accuracy,
            "avg_confidence":    self.avg_confidence,
            "per_ontology":      {k: v.to_dict() for k, v in self.per_ontology.items()},
            "per_entity":        {k: v.to_dict() for k, v in self.per_entity.items()},
        }

    def __str__(self) -> str:
        return (
            f"EvaluationMetrics("
            f"total={self.total}, "
            f"exact={self.exact} ({self.accuracy:.1%}), "
            f"wrong={self.wrong} ({self.wrong_rate:.1%}), "
            f"hallucinated={self.hallucinated} ({self.hallucination_rate:.1%}), "
            f"unmapped={self.unmapped}, "
            f"avg_conf={self.avg_confidence:.3f})"
        )


@dataclass
class EvaluationReport:
    """Complete result of one evaluation run."""

    metrics:        EvaluationMetrics
    details:        list[EvaluationDetail] = field(default_factory=list)
    benchmark_path: str | None          = None

    def to_dataframe(self) -> pd.DataFrame:
        """Return detail rows as a pandas DataFrame."""
        return pd.DataFrame([d.to_dict() for d in self.details])

    def summary(self) -> str:
        """One-line text summary."""
        return str(self.metrics)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────


class OntologyMappingEvaluator:
    """
    Evaluate OntologyMapper accuracy against a human-verified benchmark CSV.

    Args:
        mapper:            An OntologyMapper instance (must have a ``map_term()`` method).
        ground_truth_files: One or more paths to benchmark CSV files.
                            They are concatenated before evaluation.
        use_api_validation: If True (default), validate predicted codes via external APIs
                            to classify failures as WRONG vs HALLUCINATED.
                            Requires the ``requests`` package.
        validation_delay:  Seconds to sleep between validation API calls (default: 0.1).
        validator:         Supply a pre-configured ``OntologyValidator`` instance to
                           override the default one.  Useful for testing.
        loinc_fhir_url:    LOINC FHIR server URL (forwarded to OntologyValidator).
        loinc_fhir_user:   LOINC FHIR username (forwarded to OntologyValidator).
        loinc_fhir_pass:   LOINC FHIR password (forwarded to OntologyValidator).

    Example::

        from llm_ontology_mapper import OntologyMapper
        from llm_ontology_mapper.evaluator import OntologyMappingEvaluator

        mapper    = OntologyMapper(provider="ollama", model="llama3.2")
        evaluator = OntologyMappingEvaluator(
            mapper=mapper,
            ground_truth_files=["benchmark_hpo.csv", "benchmark_mondo.csv"],
        )
        report = evaluator.evaluate()
        print(report.metrics)
        df = report.to_dataframe()
    """

    def __init__(
        self,
        mapper:              Any | None       = None,
        ground_truth_files:  list[str] | None = None,
        use_api_validation:  bool                = True,
        validation_delay:    float               = 0.1,
        validator:           OntologyValidator | None = None,
        loinc_fhir_url:      str | None = None,
        loinc_fhir_user:     str | None = None,
        loinc_fhir_pass:     str | None = None,
    ) -> None:
        self.mapper             = mapper
        self.ground_truth_files = ground_truth_files or []
        self.use_api_validation = use_api_validation
        self.validation_delay   = validation_delay

        if validator is not None:
            self.validator: OntologyValidator | None = validator
        elif use_api_validation:
            self.validator = OntologyValidator(
                cache_enabled=True,
                api_timeout=10,
                loinc_fhir_url=loinc_fhir_url,
                loinc_fhir_user=loinc_fhir_user,
                loinc_fhir_pass=loinc_fhir_pass,
            )
            logger.info(
                "OntologyValidator enabled — will call external APIs to classify "
                "WRONG vs HALLUCINATED predictions."
            )
        else:
            self.validator = None
            logger.info("API validation disabled — failures will be classified as UNKNOWN.")

        self._ground_truth: pd.DataFrame | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Ground truth loading
    # ─────────────────────────────────────────────────────────────────────────

    def load_ground_truth(self) -> pd.DataFrame:
        """
        Load and merge all benchmark CSV files into a single DataFrame.

        Expected columns:
          source_variable  — source term name
          source_label     — human-readable label (optional)
          target_code      — verified CURIE(s); pipe-separated for multi-concept rows
          target_term      — verified label (optional)
          mapped_in_entities or entity_type — entity domain (optional)

        Returns the merged DataFrame and caches it on ``self._ground_truth``.
        Raises ``ValueError`` if no files are found or all are empty.
        """
        frames: list[pd.DataFrame] = []
        for path in self.ground_truth_files:
            p = Path(path)
            if not p.exists():
                logger.warning("Ground truth file not found: %s", path)
                continue
            df = pd.read_csv(p)
            logger.info("Loaded %d rows from %s", len(df), path)
            frames.append(df)

        if not frames:
            raise ValueError("No ground truth files found or all paths are missing.")

        gt = pd.concat(frames, ignore_index=True)

        # Normalise entity column — prefer mapped_in_entities (PCGL convention)
        if "mapped_in_entities" in gt.columns and "entity_type" not in gt.columns:
            gt["entity_type"] = gt["mapped_in_entities"].apply(
                lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "Unknown"
            )
        gt["entity_type"] = gt.get("entity_type", pd.Series("Unknown", index=gt.index)).fillna("Unknown")

        # Drop rows without a target_code (cannot evaluate)
        before = len(gt)
        gt = gt[gt["target_code"].notna() & (gt["target_code"].astype(str).str.strip() != "")]
        if len(gt) < before:
            logger.info("Dropped %d rows without target_code.", before - len(gt))

        # Deduplicate
        before = len(gt)
        gt = gt.drop_duplicates(subset=["source_variable", "target_code"], keep="first")  # type: ignore[call-overload]
        if len(gt) < before:
            logger.info("Removed %d duplicate rows.", before - len(gt))

        self._ground_truth = gt
        logger.info("Ground truth ready: %d rows across %d file(s).", len(gt), len(frames))
        return gt

    # ─────────────────────────────────────────────────────────────────────────
    # Public evaluation entry point
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        sample_size:              int | None   = None,
        delay_between_requests:   float           = 0.0,
        entity_type:              str | None   = None,
        source_variable_col:      str             = "source_variable",
        source_label_col:         str             = "source_label",
        target_code_col:          str             = "target_code",
        target_term_col:          str             = "target_term",
        entity_type_col:          str             = "entity_type",
    ) -> EvaluationReport:
        """
        Run the mapper on the loaded ground truth and return an EvaluationReport.

        The ground truth must be loaded first (either by calling
        ``load_ground_truth()`` or passing ``ground_truth_files`` to ``__init__``).

        Args:
            sample_size:            Random sample size (None = evaluate all rows).
            delay_between_requests: Seconds to wait between map_term() calls.
            entity_type:            Fallback entity domain when the row has none.
            source_variable_col:    Column name for the source term.
            source_label_col:       Column name for the source label.
            target_code_col:        Column name for the ground-truth CURIE(s).
            target_term_col:        Column name for the ground-truth label.
            entity_type_col:        Column name for the entity type.

        Returns:
            EvaluationReport with EvaluationMetrics and a list of EvaluationDetail.
        """
        if self.mapper is None:
            raise ValueError("mapper= must be provided in __init__ before calling evaluate().")

        if self._ground_truth is None:
            self.load_ground_truth()

        gt = self._ground_truth
        assert gt is not None  # narrowing

        if sample_size and sample_size < len(gt):
            gt = gt.sample(n=sample_size, random_state=42)
            logger.info("Sampling %d / %d rows for evaluation.", sample_size, len(self._ground_truth))  # type: ignore[arg-type]

        logger.info("Evaluating %d rows …", len(gt))
        details: list[EvaluationDetail] = []
        total_confidence = 0.0

        for _, row in gt.iterrows():
            source_variable = str(row.get(source_variable_col, "")).strip()
            source_label    = row.get(source_label_col) or None
            ground_truth_raw = str(row.get(target_code_col, "")).strip()
            ground_truth_term = row.get(target_term_col) or None
            row_entity      = row.get(entity_type_col) or entity_type or None

            if not source_variable:
                logger.warning("Skipping row with empty source_variable.")
                continue

            # Parse ground truth (pipe-separated multi-code support)
            gt_codes = [c.strip().upper() for c in ground_truth_raw.split("|") if c.strip()]

            # ── Map ───────────────────────────────────────────────────────────
            predicted_code  = "UNMAPPED"
            predicted_term  = ""
            confidence      = 0.0
            match           = MatchType.UNMAPPED
            notes: str | None = None

            try:
                result = self.mapper.map_term(
                    source_term=source_variable,
                    source_label=source_label,
                    entity_type=row_entity,
                )
                predicted_code = result.target_code
                predicted_term = result.target_term
                confidence     = result.confidence
                notes          = result.notes

                match = self._score(predicted_code, gt_codes)

            except Exception:
                logger.exception("map_term failed for %r", source_variable)
                match = MatchType.UNMAPPED
                notes = "map_term() raised an exception"

            total_confidence += confidence

            if delay_between_requests > 0:
                time.sleep(delay_between_requests)

            details.append(EvaluationDetail(
                source_variable=source_variable,
                source_label=source_label,
                entity_type=str(row_entity) if row_entity else None,
                ground_truth_codes=gt_codes,
                ground_truth_term=str(ground_truth_term) if ground_truth_term else None,
                predicted_code=predicted_code,
                predicted_term=predicted_term,
                predicted_confidence=confidence,
                match_type=match,
                notes=notes,
            ))

        metrics = self._calculate_metrics(details, total_confidence)

        if self.validator:
            stats = self.validator.get_cache_stats()
            logger.info(
                "Validation cache: %d checked, %d valid, %d invalid, %d unknown.",
                stats["cached_codes"], stats["valid_codes"],
                stats["invalid_codes"], stats["unknown_codes"],
            )

        logger.info("Evaluation complete: %s", metrics)
        return EvaluationReport(
            metrics=metrics,
            details=details,
            benchmark_path=", ".join(self.ground_truth_files) if self.ground_truth_files else None,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Scoring logic
    # ─────────────────────────────────────────────────────────────────────────

    def _score(self, predicted_code: str, gt_codes: list[str]) -> MatchType:
        """
        Classify one prediction against a (possibly multi-code) ground truth.

        Exact  → predicted code is in the ground-truth set.
        Otherwise, call the validator to determine WRONG vs HALLUCINATED.
        Falls back to UNKNOWN when the validator is unavailable.
        """
        predicted_upper = predicted_code.strip().upper()

        if predicted_upper in ("UNMAPPED", "ERROR", "NONE", ""):
            return MatchType.UNMAPPED

        if predicted_upper in gt_codes:
            return MatchType.EXACT

        # Not an exact match — validate existence to classify the failure
        if self.validator is not None:
            if self.validation_delay > 0:
                time.sleep(self.validation_delay)
            exists = self.validator.validate_code(predicted_code)
            if exists is True:
                return MatchType.WRONG
            if exists is False:
                return MatchType.HALLUCINATED
            # exists is None → API unavailable
            return MatchType.UNKNOWN

        # No validator → cannot distinguish WRONG from HALLUCINATED
        return MatchType.UNKNOWN

    # ─────────────────────────────────────────────────────────────────────────
    # Metrics calculation
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _calculate_metrics(
        details: list[EvaluationDetail],
        total_confidence: float,
    ) -> EvaluationMetrics:
        """Aggregate EvaluationDetail rows into EvaluationMetrics."""
        n = len(details)
        metrics = EvaluationMetrics(
            total=n,
            avg_confidence=total_confidence / n if n else 0.0,
        )

        for d in details:
            # ── Overall counters ──────────────────────────────────────────────
            if d.match_type == MatchType.EXACT:
                metrics.exact += 1
            elif d.match_type == MatchType.WRONG:
                metrics.wrong += 1
            elif d.match_type == MatchType.HALLUCINATED:
                metrics.hallucinated += 1
            elif d.match_type == MatchType.UNMAPPED:
                metrics.unmapped += 1
            else:
                metrics.unknown += 1

            # ── Per-ontology slice (keyed by ground-truth prefix) ─────────────
            for gt_code in d.ground_truth_codes:
                prefix = gt_code.split(":")[0] if ":" in gt_code else gt_code
                if prefix not in metrics.per_ontology:
                    metrics.per_ontology[prefix] = OntologyBreakdown()
                OntologyMappingEvaluator._update_breakdown(
                    metrics.per_ontology[prefix], d.match_type
                )

            # ── Per-entity slice ──────────────────────────────────────────────
            entity = d.entity_type or "Unknown"
            if entity not in metrics.per_entity:
                metrics.per_entity[entity] = OntologyBreakdown()
            OntologyMappingEvaluator._update_breakdown(
                metrics.per_entity[entity], d.match_type
            )

        return metrics

    @staticmethod
    def _update_breakdown(bd: OntologyBreakdown, match_type: MatchType) -> None:
        bd.total += 1
        if match_type == MatchType.EXACT:
            bd.exact += 1
        elif match_type == MatchType.WRONG:
            bd.wrong += 1
        elif match_type == MatchType.HALLUCINATED:
            bd.hallucinated += 1
        elif match_type == MatchType.UNMAPPED:
            bd.unmapped += 1


# ─────────────────────────────────────────────────────────────────────────────
# Convenience re-export (so callers don't need to import from validator)
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "MatchType",
    "EvaluationDetail",
    "OntologyBreakdown",
    "EvaluationMetrics",
    "EvaluationReport",
    "OntologyMappingEvaluator",
    "OntologyValidator",
]

