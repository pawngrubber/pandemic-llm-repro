import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pandemic_llm_repro.data_loader import PandemicDatasetLoader

def test_loader():
    pkl_path = "../PandemicLLM/data/processed_v5_4.pkl"
    if not os.path.exists(pkl_path):
        print(f"Data not found at {pkl_path}")
        return
    
    loader = PandemicDatasetLoader(pkl_path, target='t1')
    print(f"Loaded {len(loader.df)} rows")
    
    train_ds = loader.get_hf_dataset('train')
    print(f"Train dataset size: {len(train_ds)}")
    print("Example prompt:")
    print(train_ds[0]['instruction'])
    print("Example response:")
    print(train_ds[0]['output'])

if __name__ == "__main__":
    test_loader()
