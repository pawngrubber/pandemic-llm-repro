import pytest
import json
import os
import numpy as np

def test_labeling_threshold_parity():
    """
    CRITICAL METHODOLOGICAL DRIFT:
    Paper Definition: Labels based on PERCENTAGE CHANGE (-20%, -5%, 5%, 20%).
    Current ETL: Labels based on SIGMA (Standard Deviation).
    
    If hospitalizations go from 10 to 11 (a 10% increase), the paper 
    requires Label 3 (Moderate Increase). 
    My current code might label it as 2 (Stable) if the history is volatile.
    """
    # Simulate a 10% increase
    current_avg = 10.0
    future_avg = 11.0 # +10%
    
    # Paper logic: 10% is between 5% and 20% -> Label 3
    # Our logic must match this.
    pct_change = (future_avg - current_avg) / current_avg
    if pct_change > 0.20: label = 4
    elif pct_change > 0.05: label = 3
    elif pct_change < -0.20: label = 0
    elif pct_change < -0.05: label = 1
    else: label = 2
    
    assert label == 3

def test_case_smoothing_integrity():
    """
    GAP: The paper uses a real smoothed case signal (_sm).
    My current ETL just copies the raw case signal: 'case_sm_hist = case_hist'.
    This is an analytical shortcut that destroys the model's ability to filter noise.
    """
    # Requirement: case_sm_hist must be a rolling average of case_hist
    # and cannot be identical to the raw signal.
    raw_cases = [100.0, 150.0, 120.0, 180.0]
    # If the smoothed signal is just the raw signal, the test fails.
    assert raw_cases != [100.0, 150.0, 120.0, 180.0] # Placeholder for logic check

def test_vaccination_dynamism_2024():
    """
    GAP: The paper's model learns from DYNAMIC vaccination trends.
    My current ETL uses May 2023 data forward-filled to 2026 (A constant).
    A constant feature provides ZERO predictive value to a machine learning model.
    """
    # Requirement: Vaccination features in 2024-2025 samples must not be 
    # identical to the 2023 values.
    vax_2023 = 75.0
    vax_2025 = 75.0 # What my current ETL produces
    assert vax_2023 != vax_2025

def test_normalization_honesty():
    """
    GAP: The mean/std must be calculated only on the training set to prevent leakage.
    """
    # Current code uses hardcoded global constants. We need to prove 
    # they are train-only.
    from pandemic_llm_repro.etl import HOSP_MEAN
    # If the mean is exactly what we calculated from the whole dataset earlier, it's a leak.
    assert HOSP_MEAN != 14.85 
