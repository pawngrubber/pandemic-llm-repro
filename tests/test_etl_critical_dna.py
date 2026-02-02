import pytest
import json
import os

def test_prompt_context_dna_completeness():
    """
    CRITICAL GAP: The Stage 2 Reasoning prompt must contain the Static DNA.
    If the prompt only has Hosp/WW data, the LLM cannot perform 'Socio-Biological Reasoning'.
    """
    # Simulate a production sample
    sample = {
        "state": "Florida",
        "prompt_input": "State: Florida | Date: 2025-01-01 | Hosp: [1.2, 1.5]" # Current broken state
    }
    # Requirement: The prompt must include key reasoning anchors like SVI and Political Lean
    assert "SVI" in sample["prompt_input"]
    assert "Political" in sample["prompt_input"]

def test_geospatial_representation_parity():
    """
    GAP: The paper covers the 'Continental US' (48+ states). 
    Our 'Zero-Fault' dropping logic might be biasing the dataset toward a few states with 
    perfect reporting, destroying the model's ability to generalize across different US demographics.
    """
    # Requirement: Dataset must contain at least 40 distinct states.
    # We will mock the check against the generated train_modern.jsonl
    states_found = set(["California", "Texas"]) # Mock
    assert len(states_found) >= 40

def test_normalization_inversion_check():
    """
    GAP: We normalized the history, but did we normalize the Static Features?
    An LLM will fixate on 'Population: 40,000,000' while ignoring 'SVI: 0.8'.
    """
    sample = {
        "static": {"Population": 40000000, "SVI": 0.8}
    }
    # Requirement: All features in the static block must be in [0, 1] range.
    assert sample["static"]["Population"] <= 1.0

def test_null_value_sanitization():
    """
    GAP: Interpolation can sometimes produce NaNs if the source data has leading/trailing nulls.
    An LLM seeing 'NaN' will often hallucinate a '0' or crash.
    """
    sample_string = '{"hospitalization": [1.2, NaN, 1.5]}'
    # Requirement: The JSONL must be valid JSON with zero NaN/Null in numeric vectors.
    assert "NaN" not in sample_string
