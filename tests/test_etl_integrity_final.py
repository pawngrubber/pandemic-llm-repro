import pytest
import json
import os
import pandas as pd
import numpy as np

def test_label_distribution_coverage():
    """
    CRITICAL GAP: The dataset must contain examples of all 5 labels (0, 1, 2, 3, 4).
    If Label 4 (Catastrophic Surge) is missing, the model is blind to bio-threats.
    """
    file_path = "/home/paul/Documents/code/pandemic_ml_data/train_modern.jsonl"
    labels_found = set()
    with open(file_path, 'r') as f:
        for line in f:
            labels_found.add(json.loads(line)['label'])
    
    # Requirement: All labels must be represented
    for i in range(5):
        assert i in labels_found, f"Label {i} is MISSING from the dataset. The model will never learn this state."

def test_state_scaling_fairness():
    """
    GAP: Global normalization treats all states as a single population.
    Requirement: Each state's Z-Score should be centered around its own history.
    """
    file_path = "/home/paul/Documents/code/pandemic_ml_data/train_modern.jsonl"
    df = pd.read_json(file_path, lines=True)
    
    # Check CA vs WY mean
    ca_mean = np.mean([val for sublist in df[df['state'] == 'California']['hospitalization_per_100k'] for val in sublist])
    wy_mean = np.mean([val for sublist in df[df['state'] == 'Wyoming']['hospitalization_per_100k'] for val in sublist])
    
    # If CA and WY are both normalized to ~0, the difference should be small.
    # If I use Global scaling, CA will be permanently 'high' and WY 'low'.
    assert abs(ca_mean - wy_mean) < 0.5, f"Scaling Bias detected: CA mean ({ca_mean:.2f}) vs WY mean ({wy_mean:.2f})"

def test_prompt_template_completeness():
    """
    GAP: The paper requires a structured reasoning prompt.
    """
    file_path = "/home/paul/Documents/code/pandemic_ml_data/train_modern.jsonl"
    with open(file_path, 'r') as f:
        sample = json.loads(f.readline())
    
    # Requirement: The sample MUST have a 'prompt_input' key for Gemma 3.
    assert "prompt_input" in sample
    assert "Instruction:" in sample["prompt_input"]

def test_numerical_stability_inf_check():
    """
    GAP: Division by zero or small std-dev can produce INF or NAN in Z-Scores.
    """
    file_path = "/home/paul/Documents/code/pandemic_ml_data/train_modern.jsonl"
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            for val in data['hospitalization_per_100k']:
                assert np.isfinite(val), "Found INF or NAN in hospitalization vector."
