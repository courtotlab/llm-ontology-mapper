"""
OntologyMapper — core LLM mapping service.

Prompt templates live in assets/prompts/ (mapping_prompt.txt, rag_prompt.txt).
Provider backends are injected via BaseLLMProvider / LLMProviderFactory.
Public API: map_term(), map_data_dictionary().
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from llm_ontology_mapper.models import (
    LogicType,
    MappingBatch,
    MappingMetadata,
    MappingResult,
    RAGDebugInfo,
)
from llm_ontology_mapper.providers import (
    BaseLLMProvider,
    ChatMessage,
    LLMProviderFactory,
)

logger = logging.getLogger(__name__)

# ── Asset paths ───────────────────────────────────────────────────────────────
_ASSETS = Path(__file__).parent / "assets"
_PROMPT_DIR = _ASSETS / "prompts"
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
        # ── RAG parameters ────────────────────────────────────────────────────
        use_rag: bool = False,
        ontology_retriever: Any | None = None,  # OntologyRetriever
        rag_top_k: int = 5,
        rag_auto_accept_threshold: float = 0.0,
        # ── Legacy compat flags ───────────────────────────────────────────────
        use_ontogpt: bool = False,   # deprecated no-op
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
        self.ontologies = ontologies or ["HPO", "MONDO", "NCIT", "LOINC", "UO"]
        config_path = Path(ontology_config_path) if ontology_config_path else _DEFAULT_CONFIG
        self._ontology_config = self._load_config(config_path)

        # ── RAG ───────────────────────────────────────────────────────────────
        self.use_rag                  = use_rag
        self._retriever               = ontology_retriever
        self.rag_top_k                = rag_top_k
        self.rag_auto_accept_threshold = rag_auto_accept_threshold

        # ── Cache dir ─────────────────────────────────────────────────────────
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        if use_ontogpt:
            logger.warning("use_ontogpt is deprecated and will be removed in v1.0.  Ignoring.")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def map_term(
        self,
        source_term: str,
        source_label: str | None = None,
        source_type: str | None = None,
        entity_type: str | None = None,
    ) -> MappingResult:
        """
        Map a single term to an ontology code.

        Args:
            source_term:  Field name from the source data dictionary.
            source_label: Human-readable question / label text.
            source_type:  Schema type hint (radio, integer, text, …).
            entity_type:  Domain hint (phenotype, diagnosis, measurement, …).

        Returns:
            MappingResult with confidence score and logic_type.
        """
        t0 = time.monotonic()

        # 1. RAG retrieval (if enabled)
        rag_debug: RAGDebugInfo | None = None
        rag_candidates: list[dict[str, Any]] = []

        if self.use_rag and self._retriever is not None:
            rag_candidates, rag_debug = self._retrieve_candidates(
                source_term, source_label, entity_type
            )

        # 2. Build prompt
        messages = self._build_prompt(
            source_term=source_term,
            source_label=source_label,
            source_type=source_type,
            entity_type=entity_type,
            rag_candidates=rag_candidates,
        )

        # 3. Call LLM
        completion = self._llm.complete(messages, temperature=0.1, max_tokens=512)

        # 4. Parse response → MappingResult
        result = self._parse_response(
            completion=completion,
            source_term=source_term,
            source_label=source_label,
            source_type=source_type,
            rag_debug=rag_debug,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

        logger.info(
            "Mapped %r → %s (%s, conf=%.2f, logic=%s)",
            source_term, result.target_code, result.target_term,
            result.confidence, result.logic_type.value,
        )
        return result

    def map_data_dictionary(
        self,
        records: list[dict[str, Any]],
        source_term_field: str = "field_name",
        source_label_field: str = "field_label",
        source_type_field: str = "field_type",
        entity_type: str | None = None,
        study_id: str | None = None,
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

        Returns:
            MappingBatch containing one MappingResult per input record.
        """
        results: list[MappingResult] = []
        for rec in records:
            try:
                result = self.map_term(
                    source_term=rec.get(source_term_field, ""),
                    source_label=rec.get(source_label_field),
                    source_type=rec.get(source_type_field),
                    entity_type=entity_type,
                )
                results.append(result)
            except Exception:
                logger.exception("Failed to map record: %r", rec)

        return MappingBatch(study_id=study_id, entity_type=entity_type, results=results)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_config(self, path: Path) -> dict[str, Any]:
        """Load ontology_config.yaml.  Cached per path."""
        return _load_config_cached(path)

    def _load_prompt_template(self, name: str) -> str:
        """Load a prompt template from assets/prompts/<name>.txt."""
        p = _PROMPT_DIR / f"{name}.txt"
        if not p.exists():
            raise FileNotFoundError(f"Prompt template not found: {p}")
        return p.read_text(encoding="utf-8")

    def _get_ontology_description(self, ontology: str) -> str:
        return self._ontology_config.get('ontologies', {}).get(ontology, {}).get('description', ontology)

    def _get_ontology_prefix(self, ontology: str) -> str:
        return self._ontology_config.get('ontologies', {}).get(ontology, {}).get('curie_prefix', ontology)

    def get_recommended_ontologies(self, entity_type: str | None = None) -> list[str]:
        if not entity_type:
            entity_type = 'default'
        mapping = self._ontology_config.get('entity_ontology_mapping', {})
        et_lower = entity_type.lower()
        if et_lower in mapping:
            return mapping[et_lower].get('ontologies', ['HPO'])
        for key in mapping:
            if key in et_lower or et_lower in key:
                return mapping[key].get('ontologies', ['HPO'])
        return mapping.get('default', {}).get('ontologies', ['HPO', 'MONDO', 'NCIT'])

    def _infer_ontology_from_entity(self, entity_type: str | None) -> str:
        if not entity_type:
            entity_type = 'default'
        mapping = self._ontology_config.get('entity_ontology_mapping', {})
        et_lower = entity_type.lower()
        if et_lower in mapping:
            primary = mapping[et_lower].get('primary')
            if primary:
                return primary
        for key in mapping:
            if key in et_lower:
                primary = mapping[key].get('primary')
                if primary:
                    return primary
        return mapping.get('default', {}).get('primary', 'HPO')

    def _infer_ontology_source_from_code(self, code: str, fallback: str) -> str:
        if not code or ':' not in code:
            return fallback
        prefix = code.split(':', 1)[0].upper()
        for onto_key, onto_cfg in self._ontology_config.get('ontologies', {}).items():
            if onto_cfg.get('curie_prefix', onto_key).upper() == prefix:
                return onto_key
        prefix_aliases = self._ontology_config.get('prefix_aliases', {})
        if prefix in prefix_aliases:
            return prefix_aliases[prefix]
        if not prefix_aliases:
            _FB = {
                'HP': 'HPO', 'HPO': 'HPO', 'MONDO': 'MONDO', 'NCIT': 'NCIT',
                'LOINC': 'LOINC', 'ICD10': 'ICD10', 'ICD10CM': 'ICD10',
                'RXNORM': 'RxNorm', 'RXCUI': 'RxNorm',
                'SCTID': 'SNOMED-CT', 'SNOMEDCT': 'SNOMED-CT',
            }
            return _FB.get(prefix, fallback)
        return fallback

    def _normalize_ontology_code(self, code: str, target_ontology: str) -> str:
        if not code or ':' not in code:
            return code
        current_prefix, code_id = code.split(':', 1)
        current_prefix = current_prefix.strip()
        ontologies_meta = self._ontology_config.get('ontologies', {})
        correct_prefix = ontologies_meta.get(target_ontology, {}).get('curie_prefix')
        prefix_aliases = self._ontology_config.get('prefix_aliases', {})
        if not prefix_aliases:
            prefix_aliases = {
                'HPO': 'HPO', 'HP': 'HPO', 'SCTID': 'SNOMED-CT',
                'SNOMED': 'SNOMED-CT', 'SNOMEDCT': 'SNOMED-CT',
                'LOINC': 'LOINC', 'MONDO': 'MONDO', 'NCIT': 'NCIT',
                'RXNORM': 'RxNorm', 'RXCUI': 'RxNorm',
            }
        alias_ontology = prefix_aliases.get(current_prefix.upper())
        if alias_ontology and alias_ontology in ontologies_meta:
            correct_prefix = ontologies_meta[alias_ontology].get('curie_prefix', alias_ontology)
        if correct_prefix and correct_prefix != current_prefix:
            return f"{correct_prefix}:{code_id}"
        return code

    def _extract_json_from_response(self, response: str) -> str:
        import re
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if json_match:
            return json_match.group(1)
        return response.strip()

    def _build_prompt(
        self,
        source_term: str,
        source_label: str | None,
        source_type: str | None,
        entity_type: str | None,
        rag_candidates: list[dict[str, Any]],
    ) -> list[ChatMessage]:
        """Build prompt messages from assets/prompts/*.txt templates."""
        target_ontologies = self.get_recommended_ontologies(entity_type)
        ontologies_block = "\n".join(
            f"  {i+1}. {o} — {self._get_ontology_description(o)}"
            for i, o in enumerate(target_ontologies)
        )

        if rag_candidates:
            candidates_block = "\n".join(
                f"  {i+1}. {c.get('code', '?')} — {c.get('term', '?')}  (score: {c.get('score', 0):.3f})"
                for i, c in enumerate(rag_candidates)
            )
            template = self._load_prompt_template("rag_prompt")
            content = template.format(
                source_term=source_term,
                source_label=source_label or "N/A",
                source_type=source_type or "N/A",
                entity_type=entity_type or "N/A",
                candidates=candidates_block,
                ontologies=ontologies_block,
            )
        else:
            template = self._load_prompt_template("mapping_prompt")
            content = template.format(
                source_term=source_term,
                source_label=source_label or "N/A",
                source_type=source_type or "N/A",
                entity_type=entity_type or "N/A",
                ontologies=ontologies_block,
            )

        return [
            ChatMessage(role="system", content="You are a biomedical ontology expert."),
            ChatMessage(role="user", content=content),
        ]

    def _retrieve_candidates(
        self,
        source_term: str,
        source_label: str | None,
        entity_type: str | None,
    ) -> tuple[list[dict[str, Any]], RAGDebugInfo]:
        """Call retriever and build RAGDebugInfo."""
        assert self._retriever is not None, "_retrieve_candidates called without a retriever"
        ontologies = self.get_recommended_ontologies(entity_type)
        candidates, top_score = self._retriever.retrieve(
            query=source_term,
            entity_type=entity_type,
            ontologies=ontologies,
        )
        debug = RAGDebugInfo(
            query_sent=source_term,
            candidates_retrieved=candidates[: self.rag_top_k],
            top_k=getattr(self._retriever, "top_k", self.rag_top_k),
            auto_accepted=top_score >= self.rag_auto_accept_threshold,
            auto_accept_threshold=self.rag_auto_accept_threshold,
        )
        return candidates, debug

    def _parse_response(
        self,
        completion: Any,
        source_term: str,
        source_label: str | None,
        source_type: str | None,
        rag_debug: RAGDebugInfo | None,
        latency_ms: float,
    ) -> MappingResult:
        """Parse LLM completion into a MappingResult."""
        import json as _json

        if isinstance(completion, str):
            text = completion
        elif hasattr(completion, 'content'):
            text = completion.content
        else:
            text = str(completion)
        json_str = self._extract_json_from_response(text)

        try:
            data = _json.loads(json_str)
        except Exception:
            return MappingResult(
                source_term=source_term,
                source_label=source_label,
                source_type=source_type,
                target_code="UNMAPPED",
                target_term="MANUAL_REVIEW_REQUIRED",
                ontology=self.ontologies[0],
                confidence=0.0,
                logic_type=LogicType.LLM,
                notes=f"Failed to parse LLM response: {text[:200]}",
                metadata=MappingMetadata(
                    model=self._llm.model,
                    provider=self._llm.provider_name,
                    latency_ms=latency_ms,
                    prompt_tokens=None,
                    completion_tokens=None,
                    rag_debug=rag_debug,
                ),
            )

        # Handle RAG selection response (selected_rank) vs direct code response
        if "selected_rank" in data:
            rank = int(data.get("selected_rank", 0))
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")
            candidates: list[dict[str, Any]] = (
                rag_debug.candidates_retrieved if rag_debug else []
            )
            if 1 <= rank <= len(candidates):
                chosen = candidates[rank - 1]
                curie = chosen.get("code", "UNMAPPED")
                term = chosen.get("term", "")
                ontology = self._infer_ontology_source_from_code(curie, self.ontologies[0])
                logic = LogicType.RAG
            else:
                curie = "UNMAPPED"
                term = "NO_MATCH_FOUND"
                ontology = self.ontologies[0]
                logic = LogicType.RAG
            return MappingResult(
                source_term=source_term,
                source_label=source_label,
                source_type=source_type,
                target_code=curie,
                target_term=term,
                ontology=ontology,
                confidence=confidence,
                logic_type=logic,
                notes=f"RAG: {reasoning}",
                metadata=MappingMetadata(
                    model=self._llm.model,
                    provider=self._llm.provider_name,
                    latency_ms=latency_ms,
                    prompt_tokens=None,
                    completion_tokens=None,
                    rag_debug=rag_debug,
                ),
            )

        # Direct code response (also used when rag_prompt.txt returns a full code)
        raw_code = data.get("code", "UNMAPPED")
        curie = self._normalize_ontology_code(raw_code, self.ontologies[0])
        term = data.get("term", "MANUAL_REVIEW_REQUIRED")
        confidence = float(data.get("confidence", 0.5))
        reasoning = data.get("reasoning", "")
        ontology = self._infer_ontology_source_from_code(curie, self.ontologies[0])
        # Honour logic_type field from rag_prompt.txt ("rag" when LLM picked a candidate)
        _lt_str = data.get("logic_type", "")
        logic_type = LogicType.RAG if _lt_str == "rag" else LogicType.LLM
        return MappingResult(
            source_term=source_term,
            source_label=source_label,
            source_type=source_type,
            target_code=curie,
            target_term=term,
            ontology=ontology,
            confidence=confidence,
            logic_type=logic_type,
            notes=f"Mapped. {reasoning}" if reasoning else "Mapped",
            metadata=MappingMetadata(
                model=self._llm.model,
                provider=self._llm.provider_name,
                latency_ms=latency_ms,
                prompt_tokens=None,
                completion_tokens=None,
                rag_debug=rag_debug,
            ),
        )
