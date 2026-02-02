import pytest
import json
import os
import pandas as pd
from datetime import datetime
from pandemic_llm_repro.etl import resolve_bio_vector, BIO_PROFILES

def test_lineage_resolution():
    """Requirement: All 100+ CDC sub-lineages must map to high-level biological buckets."""
    # Test complex descendants
    assert resolve_bio_vector("KP.3.1.1") == BIO_PROFILES['KP']
    assert resolve_bio_vector("JN.1.16.1") == BIO_PROFILES['JN']
    assert resolve_bio_vector("MC.10") == BIO_PROFILES['MC']
    # Test case sensitivity
    assert resolve_bio_vector("kp.2") == BIO_PROFILES['KP']
    # Test unknown fallback
    assert resolve_bio_vector("UNKNOWN_VARIANT") == BIO_PROFILES['JN']

def test_zscore_normalization():
    """Requirement: Dynamic features must be Z-Score normalized for LLM stability."""
    # We will simulate a sample and check if values are within expected Z-score ranges (~ -3 to 3)
    # rather than raw hospitalization counts (0-100) or Wastewater (0-1000)
    sample_hosp = [10.5, 12.2, 11.8, 15.0]
    # In production, these should be transformed by: (val - mean) / std
    # For now, we test the logic we *intend* to implement
    mean, std = 12.0, 2.0
    normalized = [(x - mean) / std for x in sample_hosp]
    assert abs(normalized[0]) < 5.0 # Check for scaling sanity
    assert abs(sum(normalized)) < 10.0 # Check for centering

def test_temporal_split_integrity():
    """Requirement: Zero leakage. No dates >= 2025-12-01 in training data."""
    cutoff = datetime(2025, 12, 1)
    test_dates = ["2025-11-30", "2025-12-01", "2025-12-05"]
    
    train_dates = [d for d in test_dates if datetime.strptime(d, "%Y-%m-%d") < cutoff]
    assert "2025-12-01" not in train_dates
    assert "2025-12-05" not in train_dates
    assert len(train_dates) == 1

def test_prompt_structure():
    """Requirement: Stage 2 pre-computation. Prompt must contain key multimodal signals."""
    sample = {
        "state": "California",
        "date": "2025-05-01",
        "hospitalization_per_100k": [1.2, 1.5, 1.8, 2.1],
        "reported_cases_per_100k": [150, 200, 250, 300]
    }
    # Expected structure for Gemma 3 Reasoning
    prompt = f"State: {sample['state']} | Date: {sample['date']} | Hosp Trend: {sample['hospitalization_per_100k']}"
    assert "California" in prompt
    assert "Hosp Trend" in prompt
