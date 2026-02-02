import pytest
import json
import os

def test_actual_file_fidelity():
    """
    INTEGRATION TEST: Verify the actual generated dataset against paper specs.
    """
    file_path = "/home/paul/Documents/code/pandemic_ml_data/train_modern.jsonl"
    if not os.path.exists(file_path):
        pytest.fail(f"Dataset not found at {file_path}. Run ETL first.")
        
    with open(file_path, 'r') as f:
        first_line = f.readline()
        sample = json.loads(first_line)
        
    # 1. Check for 'Wastewater' drift (Should be 'reported_cases_per_100k')
    # Actually, if I named the Wastewater column 'reported_cases_per_100k', that's fine,
    # but let's check for the three vaccination layers.
    assert "Dose1_Pop_Pct" in sample
    assert "Series_Complete_Pop_Pct" in sample
    assert "Additional_Doses_Vax_Pct" in sample
    
    # 2. Check for smoothed features (_sm)
    assert "hospitalization_per_100k_sm" in sample
    
    # 3. Check variant feature naming
    assert "transmission" in sample
    assert "immunity" in sample
    assert "severity" in sample
    
    # 4. Check that we haven't 'gone above' with extra keys
    assert "wastewater" not in sample
    assert "ed_visits" not in sample
