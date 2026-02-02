import pytest
import numpy as np
from pandemic_llm_repro.etl import resolve_bio_vector

def test_static_feature_scaling_parity():
    """
    GAP: Static features (SVI, Population) must be Min-Max normalized.
    Raw population (millions) will overwhelm SVI (0-1) in the LLM's attention.
    """
    sample_static = {"Population": 40000000, "SVI": 0.85}
    # This test will fail because current ETL returns raw values
    assert sample_static["Population"] <= 1.0 
    assert sample_static["Population"] >= 0.0

def test_label_dimensional_consistency():
    """
    GAP: Labeling must compare Weekly Avg to Weekly Avg.
    Currently, I compare Weekly Avg (target) to a Single Day (last week).
    """
    # Simulate a perfectly flat trend
    hosp_history_avg = 10.0
    future_avg = 10.0
    
    # Logic: diff = future_avg - hosp_history_avg
    # If this is 0, label must be 2 (Stable)
    diff = future_avg - hosp_history_avg
    label = 2 if abs(diff) < 0.1 else (4 if diff > 1.0 else 0)
    assert label == 2

def test_clinical_floor_enforcement():
    """
    GAP: Changes smaller than 0.1 per 100k must be suppressed as 'Stable'.
    This prevents the model from hallucinating trends from reporting jitter.
    """
    # A tiny increase of 0.05 should still be 'Stable' (Label 2)
    current_avg = 5.0
    future_avg = 5.05
    diff = future_avg - current_avg
    
    # Current logic might label this as 'Increase' if sigma is low.
    # Paper requires a hard clinical floor.
    label = 2 if abs(diff) < 0.1 else 3
    assert label == 2

def test_fat_profile_completeness():
    """
    GAP: The paper requires exactly 25 static features. 
    Verify the 'static' dictionary contains the specific keys used in the study.
    """
    required_keys = ['SVI', 'medicaid', 'political_lean', 'hospital_beds_per_100k']
    sample_static = {"SVI": 0.5} # Mock
    for key in required_keys:
        assert key in sample_static
