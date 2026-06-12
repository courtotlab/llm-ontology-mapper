"""
A4 tests — AgenticMapper agent loop.

Covers:
  - normalize_code() canonicalisation
  - _select_tools() hard-filter by target_ontology
  - Happy path: sys_bp → LOINC:8480-6 (grounding pass)
  - Grounding retry: first LLM answer fails grounding; retry succeeds
  - Grounding failure: both grounding checks fail → LLM fallback logic_type
  - target_ontology prefix enforcement (HP prefix rejected when target=LOINC)
  - tool-selection hard-filter (only search_loinc offered when target=LOINC)
  - clinical_area hint injected into system prompt
  - Tool-error resilience (search_loinc raises → loop continues)
  - Max-iterations guard → UNMAPPED after MAX_ITERATIONS
  - cancel_check → immediate UNMAPPED
  - Empty response ×2 → UNMAPPED
  - Unparseable JSON ×2 → UNMAPPED
  - Alternatives populated from accumulated tool results
  - Existing suite regression (import guard)

All HTTP and LLM calls are mocked; no live services required.

Run with:  pytest tests/test_agentic_mapper.py -v -m unit
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from llm_ontology_mapper.agentic_mapper import (
    AgenticMapper,
    ALL_TOOLS,
    _TOOL_SEARCH_ICD10,
    _TOOL_SEARCH_LOINC,
    _TOOL_SEARCH_OLS,
    _TOOL_SEARCH_RXNORM,
    normalize_code,
)
from llm_ontology_mapper.models import LogicType
from llm_ontology_mapper.providers import ToolCall


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_tool_call(name: str, arguments: dict, tc_id: str = "tc_1") -> ToolCall:
    return ToolCall(id=tc_id, name=name, arguments=arguments)


def _loinc_hit(code: str = "LOINC:8480-6", term: str = "Systolic blood pressure", score: float = 0.95):
    return {"code": code, "term": term, "score": score, "definition": "", "source": "LOINC-FHIR"}


def _json_answer(
    code: str = "LOINC:8480-6",
    term: str = "Systolic blood pressure",
    ontology: str = "LOINC",
    confidence: float = 0.95,
    notes: str = "exact match",
) -> str:
    return json.dumps({
        "code": code, "term": term, "ontology": ontology,
        "confidence": confidence, "notes": notes,
    })


def _make_search_tools(
    loinc_results=None,
    ols_results=None,
    rxnorm_results=None,
    icd10_results=None,
) -> MagicMock:
    st = MagicMock()
    st.api_timeout = 10
    st.search_loinc.return_value   = loinc_results  or []
    st.search_ols.return_value     = ols_results    or []
    st.search_rxnorm.return_value  = rxnorm_results or []
    st.search_icd10.return_value   = icd10_results  or []
    return st


def _make_provider(*side_effects) -> MagicMock:
    """Return a mock provider whose complete_with_tools yields successive values."""
    prov = MagicMock()
    prov.complete_with_tools.side_effect = list(side_effects)
    return prov


# ─────────────────────────────────────────────────────────────────────────────
# normalize_code
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("raw,expected", [
    ("HP:0002110",       "HP:0002110"),         # already canonical
    ("hp:0002110",       "HP:0002110"),          # lowercase prefix → upper
    ("HP_0002110",       "HP:0002110"),          # underscore separator
    ("HP/0002110",       "HP:0002110"),          # slash separator
    ("LOINC:8480-6",     "LOINC:8480-6"),        # pass-through
    ("  MONDO:0005180 ", "MONDO:0005180"),       # strip whitespace
    ("8480-6",           "LOINC:8480-6"),        # bare local-id with source_prefix
    ("8480-6",           "8480-6"),              # bare local-id without source_prefix
    ("RXNORM:1049502",   "RXNORM:1049502"),
])
def test_normalize_code(raw: str, expected: str) -> None:
    if raw == "8480-6" and expected == "LOINC:8480-6":
        assert normalize_code(raw, source_prefix="LOINC") == expected
    elif raw == "8480-6" and expected == "8480-6":
        assert normalize_code(raw) == expected
    else:
        assert normalize_code(raw) == expected


# ─────────────────────────────────────────────────────────────────────────────
# _select_tools — hard-filter routing
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_select_tools_no_target_returns_all() -> None:
    mapper = AgenticMapper(MagicMock(), _make_search_tools())
    tools = mapper._select_tools(None)
    assert set(t["name"] for t in tools) == {
        "search_ols", "search_loinc", "search_rxnorm", "search_icd10"
    }


@pytest.mark.unit
@pytest.mark.parametrize("target,expected_names", [
    ("LOINC",      {"search_loinc"}),
    ("RXNORM",     {"search_rxnorm"}),
    ("ICD10",      {"search_icd10"}),
    ("ICD10CM",    {"search_icd10"}),
    ("HPO",        {"search_ols"}),
    ("HP",         {"search_ols"}),
    ("MONDO",      {"search_ols"}),
    ("NCIT",       {"search_ols"}),
    ("UBERON",     {"search_ols"}),
    ("CHEBI",      {"search_ols"}),
    ("SNOMED",     {"search_ols"}),
    ("SNOMEDCT",   {"search_ols"}),
])
def test_select_tools_target_ontology_routing(target: str, expected_names: set) -> None:
    mapper = AgenticMapper(MagicMock(), _make_search_tools())
    tools = mapper._select_tools(target)
    assert {t["name"] for t in tools} == expected_names


@pytest.mark.unit
def test_select_tools_unknown_ontology_returns_all() -> None:
    mapper = AgenticMapper(MagicMock(), _make_search_tools())
    tools = mapper._select_tools("CUSTOM_ONTO")
    assert len(tools) == 4


# ─────────────────────────────────────────────────────────────────────────────
# Happy path: sys_bp → LOINC:8480-6
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_happy_path_sys_bp_loinc() -> None:
    """
    Two-iteration loop: LLM calls search_loinc → accumulates LOINC:8480-6 →
    LLM returns JSON → grounding passes → AGENTIC result.
    """
    st = _make_search_tools(loinc_results=[_loinc_hit()])
    tc = _make_tool_call("search_loinc", {"query": "systolic blood pressure"})
    prov = _make_provider(
        (None,    [tc]),                               # iter 0: tool call
        (_json_answer(), []),                          # iter 1: final answer
    )

    mapper = AgenticMapper(prov, st)
    result = mapper.map("sys_bp", source_label="Systolic blood pressure", target_ontology="LOINC")

    assert result.target_code  == "LOINC:8480-6"
    assert result.logic_type   == LogicType.AGENTIC
    assert result.confidence   == pytest.approx(0.95)
    assert result.ontology     == "LOINC"


@pytest.mark.unit
def test_happy_path_result_fields() -> None:
    """Verify all canonical fields on a successful grounded mapping."""
    st = _make_search_tools(loinc_results=[_loinc_hit()])
    tc = _make_tool_call("search_loinc", {"query": "systolic blood pressure"})
    prov = _make_provider(
        (None, [tc]),
        (_json_answer(), []),
    )

    mapper = AgenticMapper(prov, st)
    result = mapper.map("sys_bp")

    assert result.source_term   == "sys_bp"
    assert result.target_code   == "LOINC:8480-6"
    assert result.target_term   == "Systolic blood pressure"
    assert result.logic_type    == LogicType.AGENTIC
    assert result.confidence    >= 0.0
    assert isinstance(result.alternatives, list)


@pytest.mark.unit
def test_hpo_result_uses_ontology_name_not_code_prefix() -> None:
    code = "HP:0012735"
    st = _make_search_tools(ols_results=[
        {
            "code": code,
            "term": "Cough",
            "score": 0.95,
            "definition": "",
            "source": "OLS",
        },
    ])
    tc = _make_tool_call("search_ols", {"query": "cough", "ontology": "HP"})
    prov = _make_provider(
        (None, [tc]),
        (_json_answer(code=code, term="Cough", ontology="HP"), []),
    )

    mapper = AgenticMapper(prov, st)
    result = mapper.map("cough", target_ontology="HPO", clinical_area="phenotype")

    assert result.target_code == code
    assert result.ontology == "HPO"
    assert result.logic_type == LogicType.AGENTIC


# ─────────────────────────────────────────────────────────────────────────────
# Grounding
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_grounding_retry_succeeds_on_second_answer() -> None:
    """
    First answer gives an invalid code (not in accumulation).
    Corrective retry message is injected.
    Second answer gives a valid grounded code → AGENTIC.
    """
    valid_code   = "LOINC:8480-6"
    invalid_code = "LOINC:9999-9"          # not in any tool result

    st = _make_search_tools(loinc_results=[_loinc_hit(valid_code)])
    tc = _make_tool_call("search_loinc", {"query": "systolic blood pressure"})

    prov = _make_provider(
        (None, [tc]),                                  # iter 0: tool call → accumulates valid_code
        (_json_answer(code=invalid_code), []),          # iter 1: bad answer → grounding fails
        (_json_answer(code=valid_code),   []),          # iter 2: retry → grounding passes
    )

    mapper = AgenticMapper(prov, st)
    result = mapper.map("sys_bp")

    assert result.target_code == valid_code
    assert result.logic_type  == LogicType.AGENTIC


@pytest.mark.unit
def test_grounding_failure_falls_back_to_llm_logic_type() -> None:
    """
    Both grounding attempts fail (model insists on an ungrounded code).
    Second failure → logic_type=LLM with 'Grounding check failed' note.
    """
    valid_code   = "LOINC:8480-6"
    invalid_code = "LOINC:9999-9"

    st = _make_search_tools(loinc_results=[_loinc_hit(valid_code)])
    tc = _make_tool_call("search_loinc", {"query": "systolic blood pressure"})

    prov = _make_provider(
        (None, [tc]),
        (_json_answer(code=invalid_code), []),  # first bad answer
        (_json_answer(code=invalid_code), []),  # second bad answer → LLM fallback
    )

    mapper = AgenticMapper(prov, st)
    result = mapper.map("sys_bp")

    assert result.logic_type == LogicType.LLM
    assert result.notes is not None
    assert "Grounding check failed" in result.notes


@pytest.mark.unit
def test_grounding_pass_code_must_be_in_accumulation() -> None:
    """Code returned by LLM must exist in accumulated tool results."""
    st = _make_search_tools(loinc_results=[_loinc_hit("LOINC:8480-6")])
    tc = _make_tool_call("search_loinc", {"query": "bp"})

    prov = _make_provider(
        (None, [tc]),
        (_json_answer("LOINC:8480-6"), []),     # grounded → AGENTIC
    )

    mapper = AgenticMapper(prov, st)
    result = mapper.map("bp")
    assert result.logic_type == LogicType.AGENTIC
    assert result.target_code == "LOINC:8480-6"


# ─────────────────────────────────────────────────────────────────────────────
# target_ontology enforcement
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_target_ontology_loinc_only_tools_offered() -> None:
    """When target_ontology=LOINC, complete_with_tools receives ONLY search_loinc."""
    st = _make_search_tools(loinc_results=[_loinc_hit()])
    tc = _make_tool_call("search_loinc", {"query": "bp"})
    prov = _make_provider(
        (None, [tc]),
        (_json_answer("LOINC:8480-6"), []),
    )

    mapper = AgenticMapper(prov, st)
    mapper.map("sys_bp", target_ontology="LOINC")

    # All calls to complete_with_tools must receive tools= containing ONLY search_loinc
    for actual_call in prov.complete_with_tools.call_args_list:
        _, kwargs = actual_call
        tools_arg = actual_call.args[1] if len(actual_call.args) > 1 else kwargs.get("tools", [])
        assert all(t["name"] == "search_loinc" for t in tools_arg), (
            f"Non-LOINC tool offered when target_ontology=LOINC: {[t['name'] for t in tools_arg]}"
        )


@pytest.mark.unit
def test_target_ontology_prefix_mismatch_triggers_grounding_retry() -> None:
    """
    target_ontology=LOINC but model returns HP:0002110 (wrong prefix).
    The grounding check must fail and a corrective message must be sent.
    On the second (valid) answer, AGENTIC is returned.
    """
    st = _make_search_tools(loinc_results=[_loinc_hit("LOINC:8480-6")])
    tc = _make_tool_call("search_loinc", {"query": "bp"})

    # First answer: HP prefix (wrong for LOINC target) — but HP:0002110 is NOT
    # in accumulation anyway, so grounding fails on two counts.
    prov = _make_provider(
        (None, [tc]),
        (_json_answer(code="HP:0002110", ontology="HPO"), []),  # bad prefix
        (_json_answer(code="LOINC:8480-6"), []),                 # correct
    )

    mapper = AgenticMapper(prov, st)
    result = mapper.map("sys_bp", target_ontology="LOINC")

    assert result.target_code == "LOINC:8480-6"
    assert result.logic_type  == LogicType.AGENTIC


# ─────────────────────────────────────────────────────────────────────────────
# clinical_area routing
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_clinical_area_hint_appears_in_system_prompt() -> None:
    """
    clinical_area should be present in the system prompt message sent to the LLM.
    """
    st = _make_search_tools(loinc_results=[_loinc_hit()])
    tc = _make_tool_call("search_loinc", {"query": "bp"})
    prov = _make_provider(
        (None, [tc]),
        (_json_answer(), []),
    )

    mapper = AgenticMapper(prov, st)
    mapper.map("sys_bp", clinical_area="measurement")

    # Extract the messages= kwarg of the first complete_with_tools call
    first_call = prov.complete_with_tools.call_args_list[0]
    messages_arg = first_call.args[0] if first_call.args else first_call.kwargs["messages"]
    system_content = next(m.content for m in messages_arg if m.role == "system")

    assert "measurement" in system_content


@pytest.mark.unit
def test_no_clinical_area_produces_clean_prompt() -> None:
    """Without clinical_area, prompt must not contain the CLINICAL AREA HINT line."""
    st = _make_search_tools(loinc_results=[_loinc_hit()])
    tc = _make_tool_call("search_loinc", {"query": "bp"})
    prov = _make_provider((None, [tc]), (_json_answer(), []))

    mapper = AgenticMapper(prov, st)
    mapper.map("sys_bp")

    first_call = prov.complete_with_tools.call_args_list[0]
    messages_arg = first_call.args[0] if first_call.args else first_call.kwargs["messages"]
    system_content = next(m.content for m in messages_arg if m.role == "system")

    assert "CLINICAL AREA HINT" not in system_content


# ─────────────────────────────────────────────────────────────────────────────
# Tool-error resilience
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_tool_error_does_not_abort_loop() -> None:
    """
    If search_loinc raises, the loop logs the error but continues.
    The next iteration (text answer) should still be processed.
    """
    st = _make_search_tools()
    st.search_loinc.side_effect = RuntimeError("LOINC server unavailable")

    tc = _make_tool_call("search_loinc", {"query": "systolic blood pressure"})

    # After the tool error, provide a valid grounded answer.
    # Grounding will fail (nothing in accumulation), but fallback to LLM is OK.
    prov = _make_provider(
        (None, [tc]),                               # iter 0: tool call → raises
        (_json_answer("LOINC:8480-6"), []),          # iter 1: text answer (not grounded)
        (_json_answer("LOINC:8480-6"), []),          # iter 2: retry still not grounded → LLM fallback
    )

    mapper = AgenticMapper(prov, st)
    result = mapper.map("sys_bp")

    # Must not raise; result has some logic_type (LLM fallback since nothing accumulated)
    assert result.target_code in ("LOINC:8480-6", "UNKNOWN:UNMAPPED")


@pytest.mark.unit
def test_tool_error_loop_continues_to_final_answer() -> None:
    """
    One bad tool + one good tool in the same turn: good tool's results are
    accumulated, and the final answer can be grounded.
    """
    st = MagicMock()
    st.api_timeout = 10
    st.search_loinc.return_value  = [_loinc_hit("LOINC:8480-6")]
    st.search_ols.side_effect     = RuntimeError("OLS down")

    tc_loinc = _make_tool_call("search_loinc", {"query": "bp"}, tc_id="tc_loinc")
    tc_ols   = _make_tool_call("search_ols",   {"query": "bp", "ontology": "HP"}, tc_id="tc_ols")

    prov = _make_provider(
        (None,    [tc_ols, tc_loinc]),     # iter 0: both tools; OLS raises, LOINC succeeds
        (_json_answer("LOINC:8480-6"), []), # iter 1: grounded answer
    )

    mapper = AgenticMapper(prov, st)
    result = mapper.map("bp")

    assert result.target_code == "LOINC:8480-6"
    assert result.logic_type  == LogicType.AGENTIC


# ─────────────────────────────────────────────────────────────────────────────
# Max-iterations guard
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_max_iterations_guard_returns_unmapped() -> None:
    """Loop exhaustion after MAX_ITERATIONS → UNMAPPED with explanatory note."""
    st = _make_search_tools(loinc_results=[])   # tool returns nothing (doesn't matter)
    tc = _make_tool_call("search_loinc", {"query": "bp"})

    # Provider always returns a tool call; loop never gets a text answer.
    prov = _make_provider(*[(None, [tc])] * (AgenticMapper.MAX_ITERATIONS + 2))

    mapper = AgenticMapper(prov, st)
    result = mapper.map("sys_bp")

    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.ontology    == "UNKNOWN"
    assert result.confidence  == pytest.approx(0.0)
    assert result.logic_type  == LogicType.AGENTIC
    # complete_with_tools called exactly MAX_ITERATIONS times
    assert prov.complete_with_tools.call_count == AgenticMapper.MAX_ITERATIONS


# ─────────────────────────────────────────────────────────────────────────────
# cancel_check
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_cancel_check_before_first_llm_call() -> None:
    """cancel_check returning True on the first iteration → UNMAPPED immediately."""
    st = _make_search_tools()
    prov = _make_provider((None, []))     # provider should never be called

    mapper = AgenticMapper(prov, st)
    result = mapper.map("sys_bp", cancel_check=lambda: True)

    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.logic_type  == LogicType.AGENTIC
    prov.complete_with_tools.assert_not_called()


@pytest.mark.unit
def test_cancel_check_mid_loop() -> None:
    """cancel_check becomes True after first iteration → exits on second check."""
    st = _make_search_tools(loinc_results=[_loinc_hit()])
    tc = _make_tool_call("search_loinc", {"query": "bp"})
    prov = _make_provider(
        (None, [tc]),           # iter 0: tool call succeeds
        # iter 1 should never be reached because cancel fires at top of iter 1
    )

    call_count = {"n": 0}

    def cancel():
        # First check (before iter 0) → False; second check (before iter 1) → True
        call_count["n"] += 1
        return call_count["n"] > 1

    mapper = AgenticMapper(prov, st)
    result = mapper.map("sys_bp", cancel_check=cancel)

    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert prov.complete_with_tools.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Empty-response and non-parseable-JSON guards
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_empty_response_twice_returns_unmapped() -> None:
    """Two consecutive empty turns (no text, no tool calls) → UNMAPPED."""
    st = _make_search_tools()
    # (None, []) = no text, no tool calls
    prov = _make_provider(
        (None, []),
        (None, []),
    )

    mapper = AgenticMapper(prov, st)
    result = mapper.map("sys_bp")

    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.logic_type  == LogicType.AGENTIC


@pytest.mark.unit
def test_empty_response_once_then_nudge_sent() -> None:
    """After a single empty turn, a nudge message is added to the conversation."""
    st = _make_search_tools(loinc_results=[_loinc_hit()])
    tc = _make_tool_call("search_loinc", {"query": "bp"})
    prov = _make_provider(
        (None, []),             # iter 0: empty → nudge
        (None, [tc]),           # iter 1: tool call
        (_json_answer(), []),   # iter 2: final answer
    )

    mapper = AgenticMapper(prov, st)
    result = mapper.map("sys_bp")

    # Should eventually succeed
    assert result.target_code != "UNKNOWN:UNMAPPED"
    # Nudge message should have been injected (4 calls: system+user, nudge,
    # tool-result, final)
    assert prov.complete_with_tools.call_count == 3


@pytest.mark.unit
def test_unparseable_json_twice_returns_unmapped() -> None:
    """Two consecutive non-JSON text turns → UNMAPPED."""
    st = _make_search_tools()
    prov = _make_provider(
        ("this is not json at all", []),
        ("still not json!!!!",      []),
    )

    mapper = AgenticMapper(prov, st)
    result = mapper.map("sys_bp")

    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.logic_type  == LogicType.AGENTIC


@pytest.mark.unit
def test_unparseable_json_once_then_corrective_message() -> None:
    """After one bad JSON, a corrective message is injected before the next call."""
    st = _make_search_tools(loinc_results=[_loinc_hit()])
    tc = _make_tool_call("search_loinc", {"query": "bp"})
    prov = _make_provider(
        ("not json",             []),   # iter 0: bad text
        (None,     [tc]),               # iter 1: tool call (after corrective message)
        (_json_answer(),         []),   # iter 2: final answer
    )

    mapper = AgenticMapper(prov, st)
    result = mapper.map("sys_bp")

    assert result.target_code != "UNKNOWN:UNMAPPED"
    assert prov.complete_with_tools.call_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# Alternatives populated from accumulated tool results
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_alternatives_populated_from_accumulated_results() -> None:
    """Accumulated codes (other than the chosen one) should appear as alternatives."""
    runner_up_code = "LOINC:8462-4"
    st = _make_search_tools(loinc_results=[
        _loinc_hit("LOINC:8480-6", score=0.95),
        {"code": runner_up_code, "term": "Diastolic blood pressure", "score": 0.80,
         "definition": "", "source": "LOINC-FHIR"},
    ])
    tc = _make_tool_call("search_loinc", {"query": "blood pressure"})
    prov = _make_provider(
        (None, [tc]),
        (_json_answer("LOINC:8480-6"), []),
    )

    mapper = AgenticMapper(prov, st)
    result = mapper.map("sys_bp")

    assert result.target_code == "LOINC:8480-6"
    alt_codes = [a.code for a in result.alternatives]
    assert runner_up_code in alt_codes


@pytest.mark.unit
def test_alternatives_source_field_is_agentic() -> None:
    """All alternatives from the agent loop must carry source='agentic'."""
    st = _make_search_tools(loinc_results=[
        _loinc_hit("LOINC:8480-6", score=0.95),
        {"code": "LOINC:8462-4", "term": "DBP", "score": 0.7, "definition": "", "source": "x"},
    ])
    tc = _make_tool_call("search_loinc", {"query": "bp"})
    prov = _make_provider(
        (None, [tc]),
        (_json_answer("LOINC:8480-6"), []),
    )

    mapper = AgenticMapper(prov, st)
    result = mapper.map("bp")

    for alt in result.alternatives:
        assert alt.source == "agentic"


# ─────────────────────────────────────────────────────────────────────────────
# MCP tool format verification
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_tools_passed_in_mcp_format_with_input_schema() -> None:
    """
    AgenticMapper passes tools to complete_with_tools in MCP JSON-Schema format
    (inputSchema key, not parameters / input_schema — that conversion happens
    inside the provider).
    """
    st = _make_search_tools(loinc_results=[_loinc_hit()])
    tc = _make_tool_call("search_loinc", {"query": "bp"})
    prov = _make_provider(
        (None, [tc]),
        (_json_answer(), []),
    )

    mapper = AgenticMapper(prov, st)
    mapper.map("sys_bp", target_ontology="LOINC")

    first_call = prov.complete_with_tools.call_args_list[0]
    tools_arg = first_call.args[1] if len(first_call.args) > 1 else first_call.kwargs["tools"]

    for tool in tools_arg:
        assert "inputSchema" in tool, (
            f"Tool {tool.get('name')} missing 'inputSchema' key; "
            f"found keys: {list(tool.keys())}"
        )
        # MCP format must NOT use 'parameters' or 'input_schema' at top level
        assert "parameters"   not in tool
        assert "input_schema" not in tool


@pytest.mark.unit
def test_all_tool_defs_have_required_mcp_keys() -> None:
    """Module-level MCP tool defs must have name, description, inputSchema."""
    for tool in ALL_TOOLS:
        assert "name"        in tool
        assert "description" in tool
        assert "inputSchema" in tool
        schema = tool["inputSchema"]
        assert schema.get("type") == "object"
        assert "properties" in schema


# ─────────────────────────────────────────────────────────────────────────────
# UNMAPPED sentinel contract
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_unmapped_result_sentinel_fields() -> None:
    """_unmapped() must produce a well-formed UNMAPPED result."""
    result = AgenticMapper._unmapped("test_term", "Test Term", "reason here")
    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.ontology    == "UNKNOWN"
    assert result.confidence  == pytest.approx(0.0)
    assert result.logic_type  == LogicType.AGENTIC
    assert result.source_term == "test_term"
    assert result.notes is not None
    assert "reason here" in result.notes


@pytest.mark.unit
def test_unmapped_result_is_valid_mapping_result() -> None:
    """UNMAPPED sentinel must satisfy the MappingResult schema (Pydantic validation)."""
    from llm_ontology_mapper.models import MappingResult
    result = AgenticMapper._unmapped("x", None, "loop exhausted")
    # If Pydantic validation passes, it's a valid MappingResult.
    assert isinstance(result, MappingResult)


# ─────────────────────────────────────────────────────────────────────────────
# LLM error resilience
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_llm_error_returns_unmapped() -> None:
    """If complete_with_tools raises, map() returns UNMAPPED instead of propagating."""
    st = _make_search_tools()
    prov = MagicMock()
    prov.complete_with_tools.side_effect = RuntimeError("LLM provider down")

    mapper = AgenticMapper(prov, st)
    result = mapper.map("sys_bp")

    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.logic_type  == LogicType.AGENTIC


# ─────────────────────────────────────────────────────────────────────────────
# Existing suite regression — confirm imports still resolve
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_public_api_exports_agentic_mapper() -> None:
    """AgenticMapper and normalize_code must be importable from the package root."""
    from llm_ontology_mapper import AgenticMapper as AM, normalize_code as nc
    assert AM is AgenticMapper
    assert nc is normalize_code


@pytest.mark.unit
def test_logic_type_agentic_exists() -> None:
    """LogicType.AGENTIC must be present (A1 contract)."""
    assert LogicType.AGENTIC == "agentic"


@pytest.mark.unit
def test_alternative_mapping_source_agentic() -> None:
    """AlternativeMapping.source='agentic' must be a valid Literal value (A1 contract)."""
    from llm_ontology_mapper.models import AlternativeMapping
    alt = AlternativeMapping(code="LOINC:8480-6", term="SBP", ontology="LOINC",
                             confidence=0.9, source="agentic")
    assert alt.source == "agentic"
