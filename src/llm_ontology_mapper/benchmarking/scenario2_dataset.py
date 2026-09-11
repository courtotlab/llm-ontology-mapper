"""
Scenario 2 (retrieval-mode ablation) dataset auditing.

Reuses llm_ontology_mapper.benchmarking.dataset.load_dataset() verbatim --
the SAME BenchmarkRow contract (source_variable -> source_term, source_label,
source_description, target_ontology hard constraint, target_code gold
CURIE(s) '|'-separated, target_term reference-only) already validated by the
preceding model-selection benchmark (scripts/run_model_benchmark.py). This
module only adds the Scenario 2 pre-flight audit (Part 1): ontology
distribution, gold-code cardinality, source_description population, and gold
namespace-vs-target_ontology consistency -- it never reinterprets the schema.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from llm_ontology_mapper.benchmarking.dataset import BenchmarkRow
from llm_ontology_mapper.ontology_identity import curie_prefix_for_ontology

# Expected gold-code CURIE prefix for each target_ontology, per the Scenario 2
# spec (Part 1): HPO->HP, MONDO->MONDO, LOINC->LOINC, ICD10->ICD10,
# SNOMED->SNOMEDCT, NCIT->NCIT, RxNorm->RXNORM. Resolved via
# ontology_identity.curie_prefix_for_ontology() (the same canonical-prefix
# table the rest of the mapper uses) rather than a hand-rolled duplicate
# table, so this audit can never silently drift from the real config.
EXPECTED_GOLD_PREFIX = curie_prefix_for_ontology


@dataclass(frozen=True)
class NamespaceViolation:
    input_row: int
    source_variable: str
    target_ontology: str
    gold_code: str
    observed_prefix: str
    expected_prefix: str | None


@dataclass(frozen=True)
class Scenario2DatasetAudit:
    """Exact, derived-from-file counts required before launching a paid run
    (Part 1). Never forces a target number -- if the workbook changes, these
    counts change with it."""

    row_count: int
    ontology_distribution: dict[str, int]
    missing_source_variable_count: int
    missing_source_label_count: int
    missing_target_ontology_count: int
    missing_target_code_count: int
    source_description_populated: int
    source_description_blank: int
    gold_cardinality_distribution: dict[int, int]  # {num_gold_codes: num_rows}
    max_gold_codes_per_row: int
    namespace_violations: list[NamespaceViolation]

    @property
    def namespaces_consistent(self) -> bool:
        return len(self.namespace_violations) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "ontology_distribution": dict(self.ontology_distribution),
            "missing_source_variable_count": self.missing_source_variable_count,
            "missing_source_label_count": self.missing_source_label_count,
            "missing_target_ontology_count": self.missing_target_ontology_count,
            "missing_target_code_count": self.missing_target_code_count,
            "source_description_populated": self.source_description_populated,
            "source_description_blank": self.source_description_blank,
            "gold_cardinality_distribution": {
                str(k): v for k, v in self.gold_cardinality_distribution.items()
            },
            "max_gold_codes_per_row": self.max_gold_codes_per_row,
            "namespaces_consistent": self.namespaces_consistent,
            "namespace_violation_count": len(self.namespace_violations),
            "namespace_violations": [
                {
                    "input_row": v.input_row,
                    "source_variable": v.source_variable,
                    "target_ontology": v.target_ontology,
                    "gold_code": v.gold_code,
                    "observed_prefix": v.observed_prefix,
                    "expected_prefix": v.expected_prefix,
                }
                for v in self.namespace_violations
            ],
        }


def _namespace_violations_for_row(row: BenchmarkRow) -> list[NamespaceViolation]:
    expected_prefix = EXPECTED_GOLD_PREFIX(row.target_ontology) or None
    violations: list[NamespaceViolation] = []
    for code in row.gold_codes:
        observed_prefix = code.split(":", 1)[0].strip().upper() if ":" in code else code.strip().upper()
        if expected_prefix is None or observed_prefix != expected_prefix:
            violations.append(
                NamespaceViolation(
                    input_row=row.input_row,
                    source_variable=row.source_variable,
                    target_ontology=row.target_ontology,
                    gold_code=code,
                    observed_prefix=observed_prefix,
                    expected_prefix=expected_prefix,
                )
            )
    return violations


def audit_dataset(rows: list[BenchmarkRow]) -> Scenario2DatasetAudit:
    """Derive every Part 1 count directly from already-loaded BenchmarkRow
    records. load_dataset() already guarantees source_variable/
    target_ontology/target_code are non-blank for every row (it raises
    BenchmarkDatasetError otherwise), so the corresponding missing_* counts
    below are always 0 for a successfully loaded dataset -- they are still
    computed (not hardcoded) so the audit is honest about what it checked."""
    ontology_counts: Counter[str] = Counter(row.target_ontology for row in rows)

    missing_source_variable = sum(1 for r in rows if not r.source_variable.strip())
    missing_source_label = sum(1 for r in rows if not r.source_label)
    missing_target_ontology = sum(1 for r in rows if not r.target_ontology.strip())
    missing_target_code = sum(1 for r in rows if not r.gold_codes)

    description_populated = sum(1 for r in rows if r.source_description)
    description_blank = sum(1 for r in rows if not r.source_description)

    gold_cardinality: Counter[int] = Counter(len(r.gold_codes) for r in rows)
    max_gold = max((len(r.gold_codes) for r in rows), default=0)

    violations: list[NamespaceViolation] = []
    for row in rows:
        violations.extend(_namespace_violations_for_row(row))

    return Scenario2DatasetAudit(
        row_count=len(rows),
        ontology_distribution=dict(ontology_counts),
        missing_source_variable_count=missing_source_variable,
        missing_source_label_count=missing_source_label,
        missing_target_ontology_count=missing_target_ontology,
        missing_target_code_count=missing_target_code,
        source_description_populated=description_populated,
        source_description_blank=description_blank,
        gold_cardinality_distribution=dict(gold_cardinality),
        max_gold_codes_per_row=max_gold,
        namespace_violations=violations,
    )
