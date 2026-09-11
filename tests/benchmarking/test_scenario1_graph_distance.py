"""
Unit tests for llm_ontology_mapper.benchmarking.scenario1_graph_distance.

Builds a tiny in-memory EFO hierarchy fixture (via EfoGraphIndex, bypassing
checksum-verified file loading) so classification logic is tested fast and
hermetically, independent of the vendored/downloaded reference files. A
separate real-data smoke test at the bottom exercises the actual downloaded
EFO v3.62.0 hierarchy when present, skipping otherwise.

Run with:  pytest tests/benchmarking/test_scenario1_graph_distance.py -v -m unit
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_ontology_mapper.benchmarking.scenario1_graph_distance import (
    RELATIONSHIP_MORE_GENERAL,
    RELATIONSHIP_MORE_SPECIFIC,
    RELATIONSHIP_NOT_APPLICABLE,
    RELATIONSHIP_SAME,
    RELATIONSHIP_SIBLING,
    RELATIONSHIP_UNRELATED,
    EfoGraphDataError,
    EfoGraphIndex,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_graph_module_state():
    """EXPECTED_SHA256/_INDEX_CACHE are module-level state that _build_index
    deliberately mutates per-test (to bypass checksum verification for small
    fixture files); reset both before every test so ordering never leaks
    state between tests."""
    import llm_ontology_mapper.benchmarking.scenario1_graph_distance as gd

    original_hashes = dict(gd.EXPECTED_SHA256)
    yield
    gd.EXPECTED_SHA256.clear()
    gd.EXPECTED_SHA256.update(original_hashes)
    gd._INDEX_CACHE.clear()  # noqa: SLF001


def _write_tsv(path: Path, header: tuple[str, str], rows: list[tuple[str, str]]) -> None:
    lines = ["\t".join(header)]
    lines.extend("\t".join(r) for r in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_index(tmp_path: Path, *, asserted: list[tuple[str, str]], entailed: list[tuple[str, str]]) -> EfoGraphIndex:
    """Build an EfoGraphIndex from small fixture edge tables.

    Fixture hierarchy used across most tests (asserted SubClassOf edges):

        EFO:A          (root-ish)
        ├── EFO:B
        │   ├── EFO:C
        │   │   └── EFO:D      (child of C, grandchild of B)
        │   └── EFO:E          (sibling of C -- shares parent EFO:B)
        └── EFO:F              (sibling of B -- shares parent EFO:A)

    `entailed` is the transitive closure implied by `asserted` (built
    explicitly per test rather than computed, so the fixture stays simple
    and each test's setup is self-evident).
    """
    edges_path = tmp_path / "efo_edges.tsv"
    entailed_path = tmp_path / "efo_entailed_edges.tsv"
    _write_tsv(edges_path, ("Subject", "Object"), asserted)
    _write_tsv(entailed_path, ("Subject", "Object"), entailed)

    # Bypass checksum verification (SHA256 pinning is tested separately) by
    # monkeypatching the expected hash to match the fixture file's actual hash.
    import llm_ontology_mapper.benchmarking.scenario1_graph_distance as gd

    gd.EXPECTED_SHA256[gd.EFO_EDGES_FILENAME] = gd._sha256(edges_path)  # noqa: SLF001
    gd.EXPECTED_SHA256[gd.EFO_ENTAILED_EDGES_FILENAME] = gd._sha256(entailed_path)  # noqa: SLF001
    return EfoGraphIndex(tmp_path)


_STANDARD_ASSERTED = [
    ("EFO:B", "EFO:A"),
    ("EFO:F", "EFO:A"),
    ("EFO:C", "EFO:B"),
    ("EFO:E", "EFO:B"),
    ("EFO:D", "EFO:C"),
]
_STANDARD_ENTAILED = [
    # direct edges are entailed too
    ("EFO:B", "EFO:A"),
    ("EFO:F", "EFO:A"),
    ("EFO:C", "EFO:B"),
    ("EFO:E", "EFO:B"),
    ("EFO:D", "EFO:C"),
    # transitive closure
    ("EFO:C", "EFO:A"),
    ("EFO:E", "EFO:A"),
    ("EFO:D", "EFO:B"),
    ("EFO:D", "EFO:A"),
]


@pytest.fixture
def standard_index(tmp_path: Path) -> EfoGraphIndex:
    return _build_index(tmp_path, asserted=_STANDARD_ASSERTED, entailed=_STANDARD_ENTAILED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. exact code -> Same
# ─────────────────────────────────────────────────────────────────────────────


def test_exact_code_is_same(standard_index: EfoGraphIndex) -> None:
    result = standard_index.classify("EFO:C", ["EFO:C"])
    assert result.graph_relationship == RELATIONSHIP_SAME
    assert result.graph_matched_gold_code == "EFO:C"


# ─────────────────────────────────────────────────────────────────────────────
# 2/3. prediction child/grandchild of gold via entailed edge -> More Specific
# ─────────────────────────────────────────────────────────────────────────────


def test_prediction_child_of_gold_is_more_specific(standard_index: EfoGraphIndex) -> None:
    # EFO:C is a direct entailed child of gold EFO:B (gold is ancestor of predicted)
    result = standard_index.classify("EFO:C", ["EFO:B"])
    assert result.graph_relationship == RELATIONSHIP_MORE_SPECIFIC
    assert result.graph_matched_gold_code == "EFO:B"


def test_prediction_grandchild_of_gold_is_more_specific(standard_index: EfoGraphIndex) -> None:
    # EFO:D is an entailed descendant of gold EFO:A (two hops up)
    result = standard_index.classify("EFO:D", ["EFO:A"])
    assert result.graph_relationship == RELATIONSHIP_MORE_SPECIFIC
    assert result.graph_matched_gold_code == "EFO:A"


# ─────────────────────────────────────────────────────────────────────────────
# 4/5. prediction parent/ancestor of gold -> More General
# ─────────────────────────────────────────────────────────────────────────────


def test_prediction_parent_of_gold_is_more_general(standard_index: EfoGraphIndex) -> None:
    result = standard_index.classify("EFO:B", ["EFO:C"])
    assert result.graph_relationship == RELATIONSHIP_MORE_GENERAL
    assert result.graph_matched_gold_code == "EFO:C"


def test_prediction_ancestor_of_gold_is_more_general(standard_index: EfoGraphIndex) -> None:
    result = standard_index.classify("EFO:A", ["EFO:D"])
    assert result.graph_relationship == RELATIONSHIP_MORE_GENERAL
    assert result.graph_matched_gold_code == "EFO:D"


# ─────────────────────────────────────────────────────────────────────────────
# 6. shared asserted direct parent -> Sibling
# ─────────────────────────────────────────────────────────────────────────────


def test_shared_asserted_parent_is_sibling(standard_index: EfoGraphIndex) -> None:
    # EFO:C and EFO:E both have asserted parent EFO:B, no ancestor/descendant relation
    result = standard_index.classify("EFO:C", ["EFO:E"])
    assert result.graph_relationship == RELATIONSHIP_SIBLING
    assert result.graph_matched_gold_code == "EFO:E"
    assert result.graph_shared_parent_code == "EFO:B"


# ─────────────────────────────────────────────────────────────────────────────
# 7. no qualifying relationship -> Unrelated
# ─────────────────────────────────────────────────────────────────────────────


def test_no_relationship_is_unrelated(tmp_path: Path) -> None:
    index = _build_index(
        tmp_path,
        asserted=[("EFO:X", "EFO:ROOT1"), ("EFO:Y", "EFO:ROOT2")],
        entailed=[("EFO:X", "EFO:ROOT1"), ("EFO:Y", "EFO:ROOT2")],
    )
    result = index.classify("EFO:X", ["EFO:Y"])
    assert result.graph_relationship == RELATIONSHIP_UNRELATED
    assert result.graph_matched_gold_code is None


# ─────────────────────────────────────────────────────────────────────────────
# 8. precedence: Same beats all others
# ─────────────────────────────────────────────────────────────────────────────


def test_same_takes_precedence_over_sibling_or_ancestor(tmp_path: Path) -> None:
    # EFO:C would also match "More General" against EFO:D and "Sibling" against
    # EFO:E if Same were not checked first -- but predicted == one of the golds.
    index = _build_index(tmp_path, asserted=_STANDARD_ASSERTED, entailed=_STANDARD_ENTAILED)
    result = index.classify("EFO:C", ["EFO:D", "EFO:C", "EFO:E"])
    assert result.graph_relationship == RELATIONSHIP_SAME


# ─────────────────────────────────────────────────────────────────────────────
# 9. precedence: More Specific / More General evaluated before Sibling
# ─────────────────────────────────────────────────────────────────────────────


def test_ancestor_relationship_takes_precedence_over_sibling(tmp_path: Path) -> None:
    # EFO:C's asserted parent is EFO:B. Gold EFO:E also has asserted parent
    # EFO:B (sibling condition holds) AND gold EFO:B is itself an entailed
    # ancestor of EFO:C (More Specific condition also holds). More Specific
    # must win because it is checked first.
    index = _build_index(tmp_path, asserted=_STANDARD_ASSERTED, entailed=_STANDARD_ENTAILED)
    result = index.classify("EFO:C", ["EFO:E", "EFO:B"])
    assert result.graph_relationship == RELATIONSHIP_MORE_SPECIFIC
    assert result.graph_matched_gold_code == "EFO:B"


# ─────────────────────────────────────────────────────────────────────────────
# 10/11/12. multiple gold codes
# ─────────────────────────────────────────────────────────────────────────────


def test_multi_gold_same_against_second_gold(standard_index: EfoGraphIndex) -> None:
    result = standard_index.classify("EFO:C", ["EFO:X_NOT_PRESENT", "EFO:C"])
    assert result.graph_relationship == RELATIONSHIP_SAME
    assert result.graph_matched_gold_code == "EFO:C"


def test_multi_gold_ancestor_relationship_against_any_gold(standard_index: EfoGraphIndex) -> None:
    # First gold unrelated, second gold (EFO:A) is an ancestor of predicted EFO:D
    result = standard_index.classify("EFO:D", ["EFO:F", "EFO:A"])
    assert result.graph_relationship == RELATIONSHIP_MORE_SPECIFIC
    assert result.graph_matched_gold_code == "EFO:A"


def test_multi_gold_sibling_relationship_against_any_gold(standard_index: EfoGraphIndex) -> None:
    # EFO:F has no ancestor/descendant relation to EFO:C, but shares no
    # parent either; EFO:E does share EFO:B with EFO:C.
    result = standard_index.classify("EFO:C", ["EFO:F", "EFO:E"])
    assert result.graph_relationship == RELATIONSHIP_SIBLING
    assert result.graph_matched_gold_code == "EFO:E"


# ─────────────────────────────────────────────────────────────────────────────
# 13/14. imported non-EFO prediction can be graph-classified; never
# auto-Unrelated purely from namespace difference
# ─────────────────────────────────────────────────────────────────────────────


def test_non_efo_prediction_is_graph_classified_via_hierarchy(tmp_path: Path) -> None:
    # UBERON:0001 is asserted+entailed child of EFO:A -- same topology as EFO:B.
    index = _build_index(
        tmp_path,
        asserted=[*_STANDARD_ASSERTED, ("UBERON:0001", "EFO:A")],
        entailed=[*_STANDARD_ENTAILED, ("UBERON:0001", "EFO:A")],
    )
    result = index.classify("UBERON:0001", ["EFO:F"])  # siblings under EFO:A
    assert result.graph_relationship == RELATIONSHIP_SIBLING
    assert result.graph_shared_parent_code == "EFO:A"


def test_non_efo_prediction_not_automatically_unrelated(tmp_path: Path) -> None:
    index = _build_index(
        tmp_path,
        asserted=[*_STANDARD_ASSERTED, ("MONDO:0001", "EFO:B")],
        entailed=[*_STANDARD_ENTAILED, ("MONDO:0001", "EFO:B"), ("MONDO:0001", "EFO:A")],
    )
    result = index.classify("MONDO:0001", ["EFO:A"])
    assert result.graph_relationship == RELATIONSHIP_MORE_SPECIFIC
    assert result.graph_relationship != RELATIONSHIP_UNRELATED


# ─────────────────────────────────────────────────────────────────────────────
# 15. missing graph node diagnostic
# ─────────────────────────────────────────────────────────────────────────────


def test_missing_node_diagnostic_flags_absent_prediction(standard_index: EfoGraphIndex) -> None:
    result = standard_index.classify("EFO:NOT_IN_HIERARCHY", ["EFO:C"])
    assert result.graph_relationship == RELATIONSHIP_UNRELATED
    assert result.graph_prediction_found is False
    assert result.graph_gold_found is True
    assert result.note is not None and "graph_prediction_found=False" in result.note


def test_missing_node_diagnostic_flags_absent_gold(standard_index: EfoGraphIndex) -> None:
    result = standard_index.classify("EFO:C", ["EFO:NOT_IN_HIERARCHY"])
    assert result.graph_relationship == RELATIONSHIP_UNRELATED
    assert result.graph_prediction_found is True
    assert result.graph_gold_found is False


def test_found_nodes_have_no_missing_node_note_when_unrelated(tmp_path: Path) -> None:
    index = _build_index(
        tmp_path,
        asserted=[("EFO:X", "EFO:ROOT1"), ("EFO:Y", "EFO:ROOT2")],
        entailed=[("EFO:X", "EFO:ROOT1"), ("EFO:Y", "EFO:ROOT2")],
    )
    result = index.classify("EFO:X", ["EFO:Y"])
    assert result.graph_prediction_found is True
    assert result.graph_gold_found is True
    assert result.note is None


# ─────────────────────────────────────────────────────────────────────────────
# 16. all five categories sum correctly
# ─────────────────────────────────────────────────────────────────────────────


def test_five_categories_sum_to_denominator(standard_index: EfoGraphIndex) -> None:
    from collections import Counter

    pairs = [
        ("EFO:C", ["EFO:C"]),  # Same
        ("EFO:C", ["EFO:B"]),  # More Specific
        ("EFO:B", ["EFO:C"]),  # More General
        ("EFO:C", ["EFO:E"]),  # Sibling
        ("EFO:F", ["EFO:C"]),  # Unrelated (no shared parent, no ancestor/descendant)
    ]
    relationships = [standard_index.classify(p, g).graph_relationship for p, g in pairs]
    counts = Counter(relationships)
    assert sum(counts.values()) == len(pairs)
    assert set(counts) == {
        RELATIONSHIP_SAME,
        RELATIONSHIP_MORE_SPECIFIC,
        RELATIONSHIP_MORE_GENERAL,
        RELATIONSHIP_SIBLING,
        RELATIONSHIP_UNRELATED,
    }


# ─────────────────────────────────────────────────────────────────────────────
# No prediction / no gold -> Not Applicable (never Unrelated)
# ─────────────────────────────────────────────────────────────────────────────


def test_no_prediction_is_not_applicable(standard_index: EfoGraphIndex) -> None:
    result = standard_index.classify(None, ["EFO:C"])
    assert result.graph_relationship == RELATIONSHIP_NOT_APPLICABLE


def test_no_gold_is_not_applicable(standard_index: EfoGraphIndex) -> None:
    result = standard_index.classify("EFO:C", [])
    assert result.graph_relationship == RELATIONSHIP_NOT_APPLICABLE


# ─────────────────────────────────────────────────────────────────────────────
# Checksum verification -- fail clearly, never silently degrade
# ─────────────────────────────────────────────────────────────────────────────


def test_missing_files_raise_clear_error(tmp_path: Path) -> None:
    with pytest.raises(EfoGraphDataError, match="missing"):
        EfoGraphIndex(tmp_path / "does_not_exist")


def test_checksum_mismatch_raises_clear_error(tmp_path: Path) -> None:
    import llm_ontology_mapper.benchmarking.scenario1_graph_distance as gd

    edges_path = tmp_path / gd.EFO_EDGES_FILENAME
    entailed_path = tmp_path / gd.EFO_ENTAILED_EDGES_FILENAME
    _write_tsv(edges_path, ("Subject", "Object"), [("A", "B")])
    _write_tsv(entailed_path, ("Subject", "Object"), [("A", "B")])
    gd.EXPECTED_SHA256[gd.EFO_EDGES_FILENAME] = "0" * 64  # deliberately wrong
    gd.EXPECTED_SHA256[gd.EFO_ENTAILED_EDGES_FILENAME] = gd._sha256(entailed_path)  # noqa: SLF001

    with pytest.raises(EfoGraphDataError, match="SHA256 mismatch"):
        EfoGraphIndex(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Real downloaded EFO v3.62.0 data smoke check (skipped if not fetched)
# ─────────────────────────────────────────────────────────────────────────────


def test_real_efo_hierarchy_classifies_known_pair_if_present() -> None:
    """query_id=3 from the OLS-EFO smoke runs: predicted EFO:0009119
    ('precursor lymphoblastic lymphoma/leukemia') vs. gold EFO:0000094
    ('B-cell acute lymphoblastic leukemia'). Audited directly against the
    downloaded edge tables: EFO:0000094 is an entailed descendant of
    EFO:0009119, so the expected classification is More General."""
    import llm_ontology_mapper.benchmarking.scenario1_graph_distance as gd

    data_dir = gd.REPO_DIR / "data" / "text2term_evaluation"
    if not (data_dir / gd.EFO_EDGES_FILENAME).exists():
        pytest.skip("EFO graph reference data not fetched in this checkout")

    # Reset any monkeypatched expected hashes from earlier tests in this module.
    gd.EXPECTED_SHA256[gd.EFO_EDGES_FILENAME] = (
        "6aa7182b70e23addb9f6d4e24bab94520bf9ff26ea26471403dd4a568689e90c"
    )
    gd.EXPECTED_SHA256[gd.EFO_ENTAILED_EDGES_FILENAME] = (
        "589ab467d24ddd22065abc75dff3aace28a9377197edf7caef201a273015d243"
    )
    gd._INDEX_CACHE.clear()  # noqa: SLF001

    index = gd.get_graph_index(data_dir)
    result = index.classify("EFO:0009119", ["EFO:0000094"])
    assert result.graph_relationship == RELATIONSHIP_MORE_GENERAL
    assert result.graph_matched_gold_code == "EFO:0000094"
