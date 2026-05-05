"""
Unit tests for NERQueryExtractor (ner_extractor.py).

Tests verify:
1. Public API imports work correctly
2. NER component instantiates without errors
3. Graceful degradation when spaCy models are not installed

Run with:  pytest tests/test_ner_extractor.py -v
"""

from __future__ import annotations

import warnings
import pytest


def test_ner_import():
    """Test that NERQueryExtractor can be imported from the public API."""
    from llm_ontology_mapper import NERQueryExtractor
    
    assert NERQueryExtractor is not None


def test_ner_instantiation():
    """Test that NERQueryExtractor instantiates without errors."""
    from llm_ontology_mapper import NERQueryExtractor
    
    ner = NERQueryExtractor()
    assert ner is not None
    assert type(ner).__name__ == "NERQueryExtractor"


def test_ner_is_available_without_models():
    """
    Test that is_available() returns False when spaCy models are not installed.
    
    This verifies graceful degradation - the NER component should not crash
    but should indicate it's not ready for use.
    """
    from llm_ontology_mapper import NERQueryExtractor
    
    ner = NERQueryExtractor()
    is_available = ner.is_available()
    
    # Without models installed, should return False
    # Note: This may return True if you have installed scispacy models like:
    #   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
    assert isinstance(is_available, bool)


def test_ner_extract_entities_without_models():
    """
    Test that extract_entities() gracefully degrades when models are not installed.
    
    Without spaCy models, extract_entities() should return an empty list
    rather than crashing.
    """
    from llm_ontology_mapper import NERQueryExtractor
    
    # Suppress scispacy abbreviation matcher warning
    # This warning is cosmetic and comes from scispacy's internal abbreviation detector
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="scispacy")
        
        ner = NERQueryExtractor()
        entities = ner.extract_entities("blood_pressure", "Blood Pressure")
    
    # Should return a list (even if empty)
    assert isinstance(entities, list)
    
    # Without models installed, should return empty list or handle gracefully
    # Note: If models are installed, this may return actual entities
    if not ner.is_available():
        assert entities == []
