"""
Named Entity Recognition (NER) extractor for biomedical entities.

This module provides NER-based query extraction using scispaCy models to identify
biomedical entities (DISEASE, CHEMICAL) from variable labels and field names.
"""

import logging
import re
import warnings

logger = logging.getLogger(__name__)


class NERQueryExtractor:
    """
    Extract biomedical entity spans from variable labels using scispaCy NER.
    
    Uses dual-model approach:
    - en_ner_bc5cdr_md: Strict DISEASE + CHEMICAL entities (primary)
    - en_core_sci_lg: Broader biomedical entities (supplementary)
    
    Features:
    - Graceful degradation (works with 0, 1, or 2 models)
    - Automatic abbreviation expansion (HTN → hypertension)
    - Smart field name normalization (snake_case → readable text)
    - Longest-first entity prioritization
    
    Example:
        >>> ner = NERQueryExtractor()
        >>> if ner.is_available():
        ...     entities = ner.extract_entities(
        ...         field_name="htn_diag_age",
        ...         field_label="Age at hypertension diagnosis"
        ...     )
        ...     print(entities)  # ['hypertension']
    """

    def __init__(self, use_abbrev_detector: bool = True):
        """
        Initialize NER extractor with scispaCy models.
        
        Args:
            use_abbrev_detector: Enable abbreviation detection pipeline (default: True)
        """
        self._nlp_bc5cdr = None
        self._nlp_sci = None
        self._available = False
        self._abbrev_store: dict = {}
        self._load_models(use_abbrev_detector)

    def _load_models(self, use_abbrev_detector: bool) -> None:
        """
        Load scispaCy NER models with graceful degradation.
        
        Attempts to load both models but succeeds if at least one loads.
        Suppresses model loading warnings for cleaner output.
        
        Args:
            use_abbrev_detector: Add AbbreviationDetector to pipeline if True
        """
        # Suppress FutureWarnings from scispacy model loading
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            
            # Try loading BC5CDR model (primary - DISEASE + CHEMICAL)
            try:
                import spacy
                self._nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md")
                if use_abbrev_detector:
                    self._add_abbrev_detector(self._nlp_bc5cdr)
                logger.info("Loaded en_ner_bc5cdr_md model")
            except (OSError, ImportError) as e:
                logger.info(
                    "en_ner_bc5cdr_md not available. "
                    "Install: pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
                )
                logger.debug(f"BC5CDR load error: {e}")
            
            # Try loading Sci-lg model (supplementary - broader coverage)
            try:
                import spacy
                self._nlp_sci = spacy.load("en_core_sci_lg")
                if use_abbrev_detector:
                    self._add_abbrev_detector(self._nlp_sci)
                logger.info("Loaded en_core_sci_lg model")
            except (OSError, ImportError) as e:
                logger.debug(
                    "en_core_sci_lg not available. "
                    "Install: pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
                )
                logger.debug(f"Sci-lg load error: {e}")
        
        # Mark as available if at least one model loaded
        self._available = (self._nlp_bc5cdr is not None or self._nlp_sci is not None)
        
        if not self._available:
            logger.warning(
                "No scispaCy NER models available. NER extraction disabled. "
                "Install scispacy and models: pip install scispacy && "
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
            )

    def _add_abbrev_detector(self, nlp):
        """
        Add AbbreviationDetector to spaCy pipeline.
        
        Args:
            nlp: spaCy Language object to modify
        """
        try:
            if "abbreviation_detector" not in nlp.pipe_names:
                nlp.add_pipe("abbreviation_detector")
        except Exception:
            logger.debug("scispacy.abbreviation not available for abbreviation detection")

    def is_available(self) -> bool:
        """
        Check if NER models are loaded.
        
        Returns:
            True if at least one scispaCy model is available, False otherwise
        """
        return self._available

    def extract_entities(
        self,
        field_name: str,
        field_label: str | None = None
    ) -> list[str]:
        """
        Extract biomedical entity spans from variable label or name.
        
        Strategy:
        1. Prioritize field_label if provided
        2. Fall back to normalized field_name if label yields no entities
        3. Convert snake_case/camelCase to readable text for field_name
        4. Deduplicate case-insensitively, preserve original casing
        5. Sort longest-first for specificity
        
        Args:
            field_name: Variable name (e.g., 'htn_diag_age')
            field_label: Human-readable label (e.g., 'Age at HTN diagnosis')
        
        Returns:
            List of entity span strings, longest first (e.g., ['hypertension']).
            Empty list if NER is unavailable or no entities found.
        
        Example:
            >>> ner.extract_entities(
            ...     field_name="htn_diag_age",
            ...     field_label="Age at hypertension diagnosis"
            ... )
            ['hypertension']
        """
        if not self._available:
            return []
        
        spans: list[tuple[str, int]] = []  # (span_text, length) for sorting
        
        # Extract from label (primary source)
        if field_label and isinstance(field_label, str) and field_label.strip():
            label_spans = self._run_ner(field_label.strip())
            spans.extend(label_spans)
        
        # Extract from field_name if label produced nothing
        if not spans and field_name and isinstance(field_name, str):
            # Convert snake_case/camelCase to readable text for NER
            readable_name = re.sub(r'[_\-]', ' ', field_name)
            readable_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', readable_name)
            name_spans = self._run_ner(readable_name.strip())
            spans.extend(name_spans)
        
        # Deduplicate by lowercased text, preserve first occurrence
        seen: set = set()
        unique: list[str] = []
        for text, _ in sorted(spans, key=lambda x: -x[1]):  # longest first
            key = text.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(text)
        
        if unique:
            logger.debug(f"NER: '{field_label or field_name}' → {unique}")
        
        return unique

    def update_abbreviation_context(self, text: str) -> None:
        """
        Process a document to learn abbreviation expansions for later use.
        
        Call this with codebook headers or any text that defines abbreviations
        in the form "long form (ABBREV)" before processing individual variables.
        
        Example:
            >>> ner.update_abbreviation_context(
            ...     "The subject's hypertension (HTN) status was recorded."
            ... )
            >>> # Now 'HTN' will be mapped to 'hypertension' in later extractions
        
        Args:
            text: Text containing parenthetical abbreviation definitions
        """
        if not text or not self._available:
            return
        
        nlp = self._nlp_bc5cdr or self._nlp_sci
        if nlp is None:
            return
        
        try:
            doc = nlp(text)
            if hasattr(doc, '_.abbreviations'):
                for abbrev in doc._.abbreviations:
                    short = abbrev.text.strip()
                    long_form = abbrev._.long_form.text.strip()
                    if short and long_form:
                        self._abbrev_store[short.lower()] = long_form
                        logger.debug(f"NER abbreviation learned: {short} → {long_form}")
        except Exception as e:
            logger.debug(f"Abbreviation context update failed: {e}")

    def _run_ner(self, text: str) -> list[tuple[str, int]]:
        """
        Run NER on text and return (span_text, length) tuples.
        
        Applies both bc5cdr (DISEASE, CHEMICAL) and sci_lg models, merging
        results. Applies abbreviation expansion from self._abbrev_store.
        
        Args:
            text: Text to process with NER
        
        Returns:
            List of (entity_text, length) tuples for sorting
        """
        results: list[tuple[str, int]] = []
        
        # Non-biomedical NLP types to always exclude regardless of model
        _NON_BIO = {
            'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL',
            'GPE', 'LOC', 'PERSON', 'ORG', 'NORP', 'FAC', 'PRODUCT',
            'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE',
        }
        
        for is_bc5cdr, nlp in [(True, self._nlp_bc5cdr), (False, self._nlp_sci)]:
            if nlp is None:
                continue
            
            try:
                doc = nlp(text)
                
                # Collect abbreviation expansions learned from this doc
                if hasattr(doc, '_.abbreviations'):
                    for abbrev in doc._.abbreviations:
                        short = abbrev.text.strip().lower()
                        long_form = abbrev._.long_form.text.strip()
                        if short and long_form:
                            self._abbrev_store[short] = long_form
                
                for ent in doc.ents:
                    # bc5cdr: strict — only DISEASE and CHEMICAL (what this model is trained for)
                    if is_bc5cdr and ent.label_ not in {'DISEASE', 'CHEMICAL'}:
                        continue
                    
                    # sci_lg: permissive — exclude only common non-biomedical NLP types
                    if not is_bc5cdr and ent.label_ in _NON_BIO:
                        continue
                    
                    span_text = ent.text.strip()
                    if not span_text or len(span_text) < 2:
                        continue
                    
                    # Expand via learned abbreviation store
                    expanded = self._abbrev_store.get(span_text.lower(), span_text)
                    results.append((expanded, len(expanded)))
            
            except Exception as e:
                logger.debug(f"NER failed on '{text[:50]}': {e}")
        
        return results
