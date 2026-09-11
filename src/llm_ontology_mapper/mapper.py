"""
OntologyMapper — core LLM mapping service.

The only supported mapping architecture is the seven-stage planned pipeline
(QueryPlanner -> RetrievalRouter -> retriever -> CandidateNormalizer ->
CandidateMerger -> LLMReranker -> MappingResultBuilder); see
planned_pipeline.PlannedPipeline. Provider backends are injected via
BaseLLMProvider / LLMProviderFactory.
Public API: map_term(), map_data_dictionary().
"""

from __future__ import annotations

import logging
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from llm_ontology_mapper.models import (
    MappingBatch,
    MappingResult,
    RetrievalMode,
)
from llm_ontology_mapper.planned_pipeline import PlannedPipeline
from llm_ontology_mapper.providers import (
    BaseLLMProvider,
    LLMProviderFactory,
)

logger = logging.getLogger(__name__)

# ── Asset paths ───────────────────────────────────────────────────────────────
_ASSETS = Path(__file__).parent / "assets"
_DEFAULT_CONFIG = _ASSETS / "ontology_config.yaml"


@lru_cache(maxsize=8)
def _load_config_cached(path: Path) -> dict[str, Any]:
    """Load ontology_config.yaml, cached at module level to avoid lru_cache memory leak on methods."""
    import yaml  # noqa: PLC0415  # type: ignore[import-untyped]

    if not path.exists():
        logger.warning("Ontology config not found at %s — using empty config", path)
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


class OntologyMapper:
    """
    LLM-powered ontology mapping service.

    Constructor
    ───────────
    Pass either a pre-built BaseLLMProvider OR the (provider, model, api_key)
    shorthand.  Never import an LLM SDK directly here.

    Example::

        # Option A — shorthand (factory builds provider internally)
        mapper = OntologyMapper(provider="openai", model="gpt-4o")

        # Option B — local Ollama (default localhost:11434)
        mapper = OntologyMapper(provider="ollama", model="llama3")

        # Option C — remote Ollama on a VM / GPU server
        mapper = OntologyMapper(
            provider="ollama",
            model="llama3",
            base_url="http://gpu-vm.internal:11434",
            api_key="bearer-token-if-protected",   # optional
        )

        # Option D — inject a pre-configured provider (easier to test/mock)
        from llm_ontology_mapper.providers import OpenAIProvider
        mapper = OntologyMapper(llm_provider=OpenAIProvider(model="gpt-4o"))
    """

    def __init__(
        self,
        # ── Shorthand provider config (used when llm_provider is None) ──────
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: str | None = None,
        # ── Pre-built provider injection (preferred for testing) ─────────────
        llm_provider: BaseLLMProvider | None = None,
        # ── Ontology scope ────────────────────────────────────────────────────
        ontologies: list[str] | None = None,
        ontology_config_path: str | None = None,
        # ── Caching ───────────────────────────────────────────────────────────
        cache_dir: str | None = ".ontology_cache",
        # ── Retrieval knobs (consumed by the planned pipeline) ────────────────
        rag_top_k: int = 15,
        # ── Planned pipeline (the only supported mapping architecture) ────────
        use_planned_pipeline: bool = True,
        retrieval_mode: RetrievalMode | str = RetrievalMode.PUBLIC,
        planned_pipeline: Any | None = None,
        max_candidates: int | None = 20,
        max_alternatives: int = 5,
        **provider_kwargs: Any,
    ) -> None:
        # ── LLM backend ───────────────────────────────────────────────────────
        if llm_provider is not None:
            self._llm = llm_provider
        else:
            self._llm = LLMProviderFactory.from_config(
                provider=provider,
                model=model,
                api_key=api_key,
                **provider_kwargs,
            )

        # ── Config ────────────────────────────────────────────────────────────
        config_path = Path(ontology_config_path) if ontology_config_path else _DEFAULT_CONFIG
        self._ontology_config = self._load_config(config_path)
        self._explicit_ontologies = self._normalize_ontology_list(ontologies)
        self.ontologies = self._explicit_ontologies or ["HPO", "MONDO", "NCIT", "LOINC", "UO"]

        self.rag_top_k = rag_top_k
        self.max_candidates = max_candidates
        self.max_alternatives = max_alternatives

        # ── Planned pipeline: the only supported mapping architecture ──────────
        # use_planned_pipeline is accepted (and defaults to True) so an active
        # caller that still passes it explicitly -- e.g. Bridge always sends
        # use_planned_pipeline=True -- keeps working unchanged. Passing False
        # is rejected rather than silently downgraded to a removed legacy path.
        if not use_planned_pipeline:
            raise ValueError(
                "use_planned_pipeline=False is no longer supported: the legacy "
                "non-planned mapping path has been removed. OntologyMapper now "
                "always maps through the seven-stage planned pipeline."
            )
        self.use_planned_pipeline = True
        self._planned_retrieval_mode = self._coerce_planned_retrieval_mode(retrieval_mode)
        self._planned_pipeline = planned_pipeline

        # ── Cache dir ─────────────────────────────────────────────────────────
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def map_term(
        self,
        source_term: str,
        source_label: str | None = None,
        source_type: str | None = None,
        entity_type: str | None = None,
        use_planned_pipeline: bool | None = None,
        retrieval_mode: RetrievalMode | str | None = None,
        *,
        source_description: str | None = None,
        strict_target_ontology: bool = False,
    ) -> MappingResult:
        """
        Map a single term to an ontology code.

        Args:
            source_term:  Field name from the source data dictionary.
            source_label: Human-readable question / label text.
            source_type:  Schema type hint (radio, integer, text, …).
            entity_type:  Domain hint (phenotype, diagnosis, measurement, …).
            use_planned_pipeline: Accepted for call-site compatibility with
                existing planned-pipeline callers (e.g. Bridge always passes
                True). There is no other mapping architecture, so passing
                False raises ValueError instead of silently doing something
                different from what the caller asked for.
            retrieval_mode: Optional per-call retrieval mode override
                (public/local/disabled).
            source_description: Optional description of the source field for
                planned-pipeline query planning.
            strict_target_ontology: When True, a returned mapping's candidate
                must belong natively to one of the requested target
                ontologies — e.g. for target_ontology="EFO", a candidate
                merely retrieved through the EFO route (native ontology HPO,
                MONDO, …) is rejected rather than accepted. Defaults to False,
                which preserves the existing ontology-specific eligibility
                rules (including EFO imported-term support) exactly.

        Returns:
            MappingResult with confidence score and logic_type.
        """
        planned_t0 = time.monotonic()
        if use_planned_pipeline is False:
            raise ValueError(
                "use_planned_pipeline=False is no longer supported: the legacy "
                "non-planned mapping path has been removed."
            )

        mode = self._coerce_planned_retrieval_mode(
            retrieval_mode if retrieval_mode is not None else self._planned_retrieval_mode
        )
        result = self._map_term_with_planned_pipeline(
            source_term=source_term,
            source_label=source_label,
            source_description=source_description,
            source_type=source_type,
            entity_type=entity_type,
            retrieval_mode=mode,
            strict_target_ontology=strict_target_ontology,
        )
        _attach_latency_ms(result, (time.monotonic() - planned_t0) * 1000)
        return result

    def map_data_dictionary(
        self,
        records: list[dict[str, Any]],
        source_term_field: str = "field_name",
        source_label_field: str = "field_label",
        source_type_field: str = "field_type",
        entity_type: str | None = None,
        study_id: str | None = None,
        *,
        source_description_field: str | None = None,
        use_planned_pipeline: bool | None = None,
        retrieval_mode: RetrievalMode | str | None = None,
        strict_target_ontology: bool = False,
    ) -> MappingBatch:
        """
        Map every row in a data dictionary to an ontology code.

        Args:
            records:           List of dicts (one per data dictionary row).
            source_term_field: Key in each record that holds the field name.
            source_label_field: Key holding the human-readable label.
            source_type_field:  Key holding the data type.
            entity_type:        Domain hint applied to all records.
            study_id:           Optional study identifier for the batch.
            source_description_field: Optional key holding a source-field description.
            use_planned_pipeline: Optional per-row override forwarded to map_term();
                see map_term for why False is rejected rather than silently accepted.
            retrieval_mode: Optional per-row planned retrieval mode forwarded to map_term().
            strict_target_ontology: Forwarded unchanged to every map_term() call in
                the batch (see OntologyMapper.map_term).

        Returns:
            MappingBatch containing one MappingResult per input record.
        """
        if use_planned_pipeline is False:
            raise ValueError(
                "use_planned_pipeline=False is no longer supported: the legacy "
                "non-planned mapping path has been removed."
            )

        results: list[MappingResult] = []
        description_key = _normalize_optional_batch_text(source_description_field)
        for rec in records:
            try:
                result = self.map_term(
                    source_term=rec.get(source_term_field, ""),
                    source_label=rec.get(source_label_field),
                    source_type=_normalize_optional_batch_text(rec.get(source_type_field)),
                    entity_type=entity_type,
                    use_planned_pipeline=use_planned_pipeline,
                    retrieval_mode=retrieval_mode,
                    source_description=(
                        _normalize_optional_batch_text(rec.get(description_key))
                        if description_key is not None
                        else None
                    ),
                    strict_target_ontology=strict_target_ontology,
                )
                results.append(result)
            except Exception:
                logger.exception("Failed to map record: %r", rec)

        return MappingBatch(study_id=study_id, entity_type=entity_type, results=results)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _coerce_planned_retrieval_mode(
        retrieval_mode: RetrievalMode | str,
    ) -> RetrievalMode:
        if isinstance(retrieval_mode, RetrievalMode):
            return retrieval_mode
        return RetrievalMode(str(retrieval_mode).lower())

    def _get_planned_pipeline(self) -> Any:
        if self._planned_pipeline is None:
            self._planned_pipeline = PlannedPipeline(provider=self._llm)
        return self._planned_pipeline

    def _planned_allowed_target_ontologies(self) -> list[str] | None:
        """
        Resolve constructor ontologies into a planned-mode allow-list.

        None means unrestricted planner behavior.  A one-item list is still
        treated as a singular target by PlannedPipeline for backward
        compatibility; multi-item lists are strict allow-lists.
        """
        return list(self._explicit_ontologies) if self._explicit_ontologies else None

    def _map_term_with_planned_pipeline(
        self,
        *,
        source_term: str,
        source_label: str | None,
        source_description: str | None,
        source_type: str | None,
        entity_type: str | None,
        retrieval_mode: RetrievalMode,
        strict_target_ontology: bool = False,
    ) -> MappingResult:
        allowed_target_ontologies = self._planned_allowed_target_ontologies()
        pipeline = self._get_planned_pipeline()
        return pipeline.map_term(
            source_term=source_term,
            source_label=source_label,
            source_description=source_description,
            source_type=source_type,
            clinical_area=entity_type,
            target_ontology=(
                allowed_target_ontologies[0]
                if allowed_target_ontologies and len(allowed_target_ontologies) == 1
                else None
            ),
            allowed_target_ontologies=allowed_target_ontologies,
            retrieval_mode=retrieval_mode,
            max_results_per_query=self.rag_top_k,
            max_candidates=self.max_candidates,
            max_alternatives=self.max_alternatives,
            strict_target_ontology=strict_target_ontology,
        )

    def _load_config(self, path: Path) -> dict[str, Any]:
        """Load ontology_config.yaml.  Cached per path."""
        return _load_config_cached(path)

    def _normalize_ontology_list(self, ontologies: Any | None) -> list[str] | None:
        if ontologies is None:
            return None
        raw_values = [ontologies] if isinstance(ontologies, str) else list(ontologies)
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in raw_values:
            ontology = self._normalize_ontology_name(raw)
            if ontology and ontology not in seen:
                normalized.append(ontology)
                seen.add(ontology)
        return normalized or None

    def _normalize_ontology_name(self, ontology: Any) -> str | None:
        text = str(ontology or "").strip()
        if not text:
            return None

        upper = text.upper()
        aliases = self._ontology_config.get("prefix_aliases", {})
        alias_map = {str(k).upper(): str(v).strip() for k, v in aliases.items()}
        alias_value = alias_map.get(upper)
        if alias_value:
            return alias_value.upper()

        ontology_keys = self._ontology_config.get("ontologies", {})
        for key in ontology_keys:
            if str(key).upper() == upper:
                return str(key).upper()

        return upper


def _normalize_optional_batch_text(value: Any) -> str | None:
    """Normalize optional spreadsheet-like batch metadata to prompt-safe text."""
    if _is_missing_batch_value(value):
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.lower() in {"nan", "none", "null", "n/a"}:
        return None
    return text


def _is_missing_batch_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True

    try:
        not_equal_to_self = value != value
    except Exception:
        not_equal_to_self = False
    try:
        if bool(not_equal_to_self):
            return True
    except Exception:
        pass

    value_type = type(value)
    module = value_type.__module__
    name = value_type.__name__
    return module.startswith("pandas.") and (
        name in {"NAType", "NaTType"} or str(value) in {"<NA>", "NaT"}
    )


def _attach_latency_ms(result: MappingResult, latency_ms: float) -> None:
    metadata = result.metadata
    if metadata is None:
        return
    result.metadata = metadata.model_copy(update={"latency_ms": latency_ms})
