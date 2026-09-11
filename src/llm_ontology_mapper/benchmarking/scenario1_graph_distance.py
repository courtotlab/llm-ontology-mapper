"""
Scenario 1 graph-distance classification: an exact reimplementation of the
graph-comparison semantics in text2term-evaluation's
`compare_ontology_mappings.compare_mappings()`, adapted to run over our own
saved Scenario 1 predictions (predicted top-1 code + acceptable gold codes)
instead of text2term's own mapping-execution output. We do NOT execute or
import text2term's mapper -- only its graph-comparison *logic* is
reimplemented here, audited line-for-line against the published source.

Reference
─────────
    Repository: https://github.com/rsgoncalves/text2term-evaluation
    File:       compare_ontology_mappings.py, function compare_mappings()
    Pinned commit: b999dbb670fa13c9ceb1ba631a7abc7557f3293b (2024-04-25)
    EFO version used by that repository's comparison: v3.62.0
        http://www.ebi.ac.uk/efo/releases/v3.62.0/efo.owl

Audited reference logic (compare_mappings, lines ~166-227 of the pinned
file) -- `t2t_trait` is the predicted CURIE, `benchmark_traits` is the
acceptable gold CURIE set:

    t2t_trait_asserted_parents   = edges_df[Subject == t2t_trait].Object
    benchmark_traits_asserted_parents = union over g in benchmark_traits of
                                        edges_df[Subject == g].Object
    t2t_trait_parents  (ancestors)  = entailed_edges_df[Subject == t2t_trait].Object
    t2t_trait_children (descendants) = entailed_edges_df[Object == t2t_trait].Subject

    if t2t_trait in benchmark_traits:                                   -> Same
    elif any(g in t2t_trait_parents for g in benchmark_traits):         -> More Specific
    elif any(g in t2t_trait_children for g in benchmark_traits):        -> More General
    elif any(g in t2t_trait_asserted_parents                            -> Sibling
             for g in benchmark_traits_asserted_parents):
    else:                                                                -> Unrelated

This priority order (Same, More Specific, More General, Sibling, Unrelated)
is preserved exactly. `edges_df`/`entailed_edges_df` use two columns,
Subject/Object, meaning "Subject SubClassOf Object" (asserted vs. entailed
transitive closure respectively) -- so "ancestors of X" = entailed edges
where Subject==X, and "descendants of X" = entailed edges where Object==X.

The reference implementation does not special-case CURIEs absent from the
hierarchy tables -- an absent node simply has empty parent/ancestor/
descendant sets and therefore falls through to "Unrelated". We preserve that
exact behavior for the primary classification, but additionally record
`graph_prediction_found` / `graph_gold_found` diagnostics so absent-node
"Unrelated" results can be told apart from genuine topological unrelatedness
(Part 7 of the graph-distance task).

Non-EFO CURIEs (CL:, UBERON:, MONDO:, HP:, CHEBI:, ORDO:, ...) are NOT
special-cased either: the edge tables include cross-ontology asserted/
entailed SubClassOf edges wherever EFO imports or aligns with those
ontologies, so an imported concept is classified via the exact same
lookup as a native EFO concept. Namespace is never consulted.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

REPO_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = Path(
    os.environ.get("EFO_GRAPH_DATA_DIR", str(REPO_DIR / "data" / "text2term_evaluation"))
)

SOURCE_REPOSITORY = "https://github.com/rsgoncalves/text2term-evaluation"
SOURCE_FILE = "compare_ontology_mappings.py (compare_mappings)"
PINNED_COMMIT = "b999dbb670fa13c9ceb1ba631a7abc7557f3293b"
EFO_VERSION = "3.62.0"
EFO_URL = "http://www.ebi.ac.uk/efo/releases/v3.62.0/efo.owl"

EFO_EDGES_FILENAME = "efo_edges.tsv"
EFO_ENTAILED_EDGES_FILENAME = "efo_entailed_edges.tsv"

EXPECTED_SHA256: dict[str, str] = {
    EFO_EDGES_FILENAME: "6aa7182b70e23addb9f6d4e24bab94520bf9ff26ea26471403dd4a568689e90c",
    EFO_ENTAILED_EDGES_FILENAME: "589ab467d24ddd22065abc75dff3aace28a9377197edf7caef201a273015d243",
}

RELATIONSHIP_SAME = "Same"
RELATIONSHIP_MORE_SPECIFIC = "More Specific"
RELATIONSHIP_MORE_GENERAL = "More General"
RELATIONSHIP_SIBLING = "Sibling"
RELATIONSHIP_UNRELATED = "Unrelated"
RELATIONSHIP_NOT_APPLICABLE = "Not Applicable"  # no prediction and/or no gold codes to compare

ALL_RELATIONSHIPS: tuple[str, ...] = (
    RELATIONSHIP_SAME,
    RELATIONSHIP_MORE_SPECIFIC,
    RELATIONSHIP_MORE_GENERAL,
    RELATIONSHIP_SIBLING,
    RELATIONSHIP_UNRELATED,
)


class EfoGraphDataError(RuntimeError):
    """Raised when the EFO hierarchy reference data is missing or fails
    checksum verification. Callers must let this propagate -- never catch it
    and silently substitute a NOT_EVALUATED/guessed classification."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, filename: str) -> str:
    if not path.exists():
        raise EfoGraphDataError(
            f"Required EFO graph reference file is missing: {path}\n"
            "Fetch it with: uv run python scripts/fetch_efo_graph_reference_data.py"
        )
    actual = _sha256(path)
    expected = EXPECTED_SHA256[filename]
    if actual != expected:
        raise EfoGraphDataError(
            f"{path} SHA256 mismatch: expected {expected}, got {actual}. This file "
            "may be corrupted, incomplete, or a different EFO release than "
            f"v{EFO_VERSION}. Re-run: "
            "uv run python scripts/fetch_efo_graph_reference_data.py --force"
        )
    return actual


@dataclass(frozen=True)
class GraphDistanceResult:
    predicted_code: str | None
    gold_codes: tuple[str, ...]
    graph_relationship: str
    graph_matched_gold_code: str | None
    graph_shared_parent_code: str | None = None
    graph_prediction_found: bool | None = None
    graph_gold_found: bool | None = None
    note: str | None = None


def _group_to_frozensets(
    df: pd.DataFrame, key_col: str, value_col: str
) -> dict[str, frozenset[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for key, value in zip(df[key_col].to_numpy(), df[value_col].to_numpy(), strict=True):
        grouped[key].add(value)
    return {k: frozenset(v) for k, v in grouped.items()}


class EfoGraphIndex:
    """Indexed EFO SubClassOf hierarchy (asserted + entailed), built once
    from the checksum-verified text2term-evaluation edge tables so that
    per-query classification is O(1)/O(|gold|) dict/set lookups rather than
    repeated pandas scans (Part 8)."""

    def __init__(self, data_dir: Path) -> None:
        edges_path = data_dir / EFO_EDGES_FILENAME
        entailed_path = data_dir / EFO_ENTAILED_EDGES_FILENAME
        self.edges_path = edges_path
        self.entailed_path = entailed_path
        self.edges_sha256 = _verify_file(edges_path, EFO_EDGES_FILENAME)
        self.entailed_sha256 = _verify_file(entailed_path, EFO_ENTAILED_EDGES_FILENAME)

        edges_df = pd.read_csv(edges_path, sep="\t", dtype=str)
        entailed_df = pd.read_csv(entailed_path, sep="\t", dtype=str)

        # asserted direct parents: child -> {parents}  (Subject SubClassOf Object)
        self._asserted_parents = _group_to_frozensets(edges_df, "Subject", "Object")
        # entailed ancestors: descendant -> {all ancestors}
        self._entailed_ancestors = _group_to_frozensets(entailed_df, "Subject", "Object")
        # entailed descendants: ancestor -> {all descendants}
        self._entailed_descendants = _group_to_frozensets(entailed_df, "Object", "Subject")

        self._known_nodes: frozenset[str] = (
            frozenset(edges_df["Subject"])
            | frozenset(edges_df["Object"])
            | frozenset(entailed_df["Subject"])
            | frozenset(entailed_df["Object"])
        )

    def is_known(self, code: str) -> bool:
        return code in self._known_nodes

    def asserted_parents(self, code: str) -> frozenset[str]:
        return self._asserted_parents.get(code, frozenset())

    def entailed_ancestors(self, code: str) -> frozenset[str]:
        return self._entailed_ancestors.get(code, frozenset())

    def entailed_descendants(self, code: str) -> frozenset[str]:
        return self._entailed_descendants.get(code, frozenset())

    def classify(self, predicted_code: str | None, gold_codes: list[str]) -> GraphDistanceResult:
        """Classify `predicted_code` (Top-1 only -- alternatives 2-5 are not
        used here, matching the reference methodology) against the
        acceptable `gold_codes` set. Priority order: Same, More Specific,
        More General, Sibling, Unrelated (audited above) -- preserved
        exactly, including its "absent node falls through to Unrelated"
        behavior."""
        gold_tuple = tuple(gold_codes)
        if not predicted_code or not gold_tuple:
            return GraphDistanceResult(
                predicted_code=predicted_code,
                gold_codes=gold_tuple,
                graph_relationship=RELATIONSHIP_NOT_APPLICABLE,
                graph_matched_gold_code=None,
                note="No prediction and/or no gold codes to compare.",
            )

        prediction_found = self.is_known(predicted_code)
        any_gold_found = any(self.is_known(g) for g in gold_tuple)

        # A. Same -- predicted CURIE exactly equals any acceptable gold CURIE
        if predicted_code in gold_tuple:
            return GraphDistanceResult(
                predicted_code=predicted_code,
                gold_codes=gold_tuple,
                graph_relationship=RELATIONSHIP_SAME,
                graph_matched_gold_code=predicted_code,
                graph_prediction_found=prediction_found,
                graph_gold_found=any_gold_found,
            )

        # B. More Specific -- any gold is an entailed ancestor of predicted
        predicted_ancestors = self.entailed_ancestors(predicted_code)
        for gold in gold_tuple:
            if gold in predicted_ancestors:
                return GraphDistanceResult(
                    predicted_code=predicted_code,
                    gold_codes=gold_tuple,
                    graph_relationship=RELATIONSHIP_MORE_SPECIFIC,
                    graph_matched_gold_code=gold,
                    graph_prediction_found=prediction_found,
                    graph_gold_found=any_gold_found,
                )

        # C. More General -- any gold is an entailed descendant of predicted
        predicted_descendants = self.entailed_descendants(predicted_code)
        for gold in gold_tuple:
            if gold in predicted_descendants:
                return GraphDistanceResult(
                    predicted_code=predicted_code,
                    gold_codes=gold_tuple,
                    graph_relationship=RELATIONSHIP_MORE_GENERAL,
                    graph_matched_gold_code=gold,
                    graph_prediction_found=prediction_found,
                    graph_gold_found=any_gold_found,
                )

        # D. Sibling -- predicted and some gold share an asserted direct parent
        predicted_parents = self.asserted_parents(predicted_code)
        for gold in gold_tuple:
            shared = predicted_parents & self.asserted_parents(gold)
            if shared:
                return GraphDistanceResult(
                    predicted_code=predicted_code,
                    gold_codes=gold_tuple,
                    graph_relationship=RELATIONSHIP_SIBLING,
                    graph_matched_gold_code=gold,
                    graph_shared_parent_code=sorted(shared)[0],
                    graph_prediction_found=prediction_found,
                    graph_gold_found=any_gold_found,
                )

        # E. Unrelated -- preserve the reference comparator's exact result
        # (including for absent-node pairs); the diagnostic fields let
        # downstream reporting tell an absent-node "Unrelated" apart from a
        # genuine topological one without changing the classification.
        note = None
        if not prediction_found or not any_gold_found:
            note = (
                f"graph_prediction_found={prediction_found}, graph_gold_found={any_gold_found}: "
                "at least one side is absent from the EFO v3.62.0 hierarchy tables. The "
                "reference comparator still classifies this as 'Unrelated' (empty "
                "parent/ancestor/descendant sets never match), so that result is preserved "
                "here -- this note flags it as an untestable pair rather than a confirmed "
                "topological distance."
            )
        return GraphDistanceResult(
            predicted_code=predicted_code,
            gold_codes=gold_tuple,
            graph_relationship=RELATIONSHIP_UNRELATED,
            graph_matched_gold_code=None,
            graph_prediction_found=prediction_found,
            graph_gold_found=any_gold_found,
            note=note,
        )


_INDEX_CACHE: dict[Path, EfoGraphIndex] = {}


def get_graph_index(data_dir: Path | str | None = None) -> EfoGraphIndex:
    """Lazily build (and cache) the EFO graph index. Raises EfoGraphDataError
    if the reference files are missing or fail checksum verification --
    never silently returns a degraded/unavailable index."""
    resolved = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if resolved not in _INDEX_CACHE:
        _INDEX_CACHE[resolved] = EfoGraphIndex(resolved)
    return _INDEX_CACHE[resolved]


def graph_data_available(data_dir: Path | str | None = None) -> bool:
    """Best-effort, non-raising check for diagnostics/CLI messaging only.
    Actual classification must call classify()/get_graph_index() directly so
    a real failure propagates instead of being swallowed here."""
    try:
        get_graph_index(data_dir)
    except EfoGraphDataError:
        return False
    return True


def classify(
    predicted_code: str | None,
    gold_codes: list[str],
    *,
    data_dir: Path | str | None = None,
) -> GraphDistanceResult:
    """Classify one Top-1 prediction against its acceptable gold code(s)
    using the real EFO v3.62.0 hierarchy (Part 3: alternatives 2-5 are never
    consulted here). Raises EfoGraphDataError if the reference hierarchy
    data is unavailable -- callers must not catch this and fall back to a
    fabricated NOT_EVALUATED result."""
    return get_graph_index(data_dir).classify(predicted_code, gold_codes)
