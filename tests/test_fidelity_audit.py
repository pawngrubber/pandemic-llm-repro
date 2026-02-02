import pytest
import json
import numpy as np

def test_vax_signal_diversity():
    """
    CRITICAL FIDELITY GAP: The 3 vax layers must NOT be identical.
    The paper relies on the delta between Dose1 and Additional_Doses.
    """
    file_path = "/home/paul/Documents/code/pandemic_ml_data/train_modern.jsonl"
    with open(file_path, 'r') as f:
        sample = json.loads(f.readline())
    
    d1 = sample['Dose1_Pop_Pct']
    sc = sample['Series_Complete_Pop_Pct']
    ad = sample['Additional_Doses_Vax_Pct']
    
    # If they are all equal, we are failing the paper's feature architecture.
    assert not (d1 == sc == ad), "Vax Fraud: All 3 vaccination layers are identical. We've lost the booster-gap signal."

def test_smoothing_diversity():
    """
    CRITICAL FIDELITY GAP: Smoothed signals (_sm) must be different from Raw signals.
    """
    file_path = "/home/paul/Documents/code/pandemic_ml_data/train_modern.jsonl"
    with open(file_path, 'r') as f:
        sample = json.loads(f.readline())
    
    raw = sample['hospitalization_per_100k']
    sm = sample['hospitalization_per_100k_sm']
    
    # If raw == sm, the model cannot distinguish between reporting noise and true trends.
    assert not (raw == sm), "Smoothing Fraud: Raw and Smoothed signals are identical."

def test_normalization_integrity():
    """
    GAP: Verify that we aren't just using the global mean but that the math 
    is actually applied per-feature.
    """
    file_path = "/home/paul/Documents/code/pandemic_ml_data/train_modern.jsonl"
    with open(file_path, 'r') as f:
        sample = json.loads(f.readline())
        
    hosp = sample['hospitalization_per_100k']
    # If the standard deviation of our normalized features is 0 or 1.0 exactly 
    # across a single sample, it might indicate a scaling error.
    assert np.std(hosp) > 0, "Numerical Error: Hospitalization vector has no variance."
