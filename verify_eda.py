import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from data.loader import DatasetLoader
from eda import EDAOrchestrator

def main():
    # Setup paths
    data_dir = Path("data/raw")
    reports_dir = Path("reports")
    
    print("Loading data...")
    loader = DatasetLoader(data_dir)
    df = loader.load_csv("application_train.csv")
    
    # Use a sample for EDA verification to keep it fast
    df_sample = df.sample(min(20000, len(df)), random_state=42)
    
    print("Running EDA Orchestrator...")
    # Target is TARGET in this dataset
    orchestrator = EDAOrchestrator(output_dir=reports_dir, target_column="TARGET")
    
    # Run EDA on a subset of features for verification
    num_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    cat_cols = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE"]
    
    orchestrator.run(df_sample, num_features=num_cols, cat_features=cat_cols)
    
    print(f"EDA complete. Results saved to {reports_dir}")

if __name__ == "__main__":
    main()
