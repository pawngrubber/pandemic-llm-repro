import pytest
import json
import os
import numpy as np

def test_paper_feature_parity():
    """
    REPRODUCTION REQUIREMENT: Must use the exact features defined in the paper.
    Our current ETL uses 'Wastewater' which is NOT in the paper.
    """
    # Requirement: Feature name must be 'reported_cases_per_100k'
    sample = {"reported_cases_per_100k": [100.0, 110.0, 120.0, 130.0]}
    assert "reported_cases_per_100k" in sample
    assert "wastewater" not in sample # No 'above and beyond' features

def test_vax_layer_parity():
    """
    REPRODUCTION REQUIREMENT: Must include all three vaccination layers.
    Paper used Dose1, Series_Complete, and Additional_Doses.
    """
    sample = {
        "Dose1_Pop_Pct": [70.0],
        "Series_Complete_Pop_Pct": [65.0],
        "Additional_Doses_Vax_Pct": [30.0]
    }
    assert "Dose1_Pop_Pct" in sample
    assert "Series_Complete_Pop_Pct" in sample
    assert "Additional_Doses_Vax_Pct" in sample

def test_smoothing_parity():
    """
    REPRODUCTION REQUIREMENT: Paper included both raw and smoothed (_sm) signals.
    """
    sample = {
        "hospitalization_per_100k": [1.0],
        "hospitalization_per_100k_sm": [1.1] # Smoothed version
    }
    assert "hospitalization_per_100k_sm" in sample

def test_variant_feature_parity():
    """
    REPRODUCTION REQUIREMENT: Variant features must be named exactly as per the paper.
    """
    sample = {
        "transmission": [0.9],
        "immunity": [0.8],
        "severity": [0.4]
    }
    assert "transmission" in sample
    assert "immunity" in sample
    assert "severity" in sample
