"""Data preprocessing execution orchestrator script.

This script is the main entry point to clean and preprocess the raw Home Credit 
Default Risk dataset (`application_train.csv`). It performs an 80/20 train/validation 
stratified split, loads Tree-Based, Linear, and Neural Network YAML configurations, 
and executes the preprocessing pipeline for each configuration.

The output tables are saved in `data/processed/` and the reports in `reports/` 
to facilitate fast model baseline training and comparison.
"""

import sys
from pathlib import Path
import pandas as pd
import yaml
import time
from sklearn.model_selection import train_test_split

# Add src/ to the Python path to ensure module imports resolve correctly
sys.path.append(str(Path.cwd() / "src"))

from preprocessing.pipeline import PreprocessingPipeline

def run_pipeline_for_config(config_path: Path, 
                            df_train: pd.DataFrame, 
                            df_val: pd.DataFrame, 
                            output_dir: Path, 
                            reports_dir: Path, 
                            target_col: str):
    """Loads a YAML configuration, runs the pipeline, and saves processed datasets.

    Why: Splitting the dataset first and then calling fit_transform on train and 
    transform on validation ensures that scaling parameters, target mappings, 
    and outlier boundaries are learned ONLY from the training fold, strictly 
    preventing any target or data leakage.

    Args:
        config_path (Path): Path to the YAML configuration file.
        df_train (pd.DataFrame): Training DataFrame including the target column.
        df_val (pd.DataFrame): Validation DataFrame including the target column.
        output_dir (Path): Output directory for the processed CSV files.
        reports_dir (Path): Output directory for JSON reports.
        target_col (str): The name of the target column in the data.
    """
    pipeline_name = config_path.stem
    print(f"\n==========================================")
    print(f"Running Preprocessing Pipeline: {pipeline_name}")
    print(f"==========================================")
    
    # 1. Load the configuration dictionary
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # 2. Setup the output directories for report isolation
    reports_subdir = reports_dir / pipeline_name
    reports_subdir.mkdir(parents=True, exist_ok=True)
    
    pipeline = PreprocessingPipeline(config, reports_dir=str(reports_subdir), target_column=target_col)
    
    # 3. Fit and Transform the Training Set
    print("Fitting and transforming train set...")
    start_time = time.time()
    
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    
    # Re-attach target column matching the row index alignment outputted by DataCleaner
    df_train_processed = X_train_processed.copy()
    df_train_processed[target_col] = y_train.loc[X_train_processed.index]
    
    print(f"Train preprocessing complete in {time.time() - start_time:.2f} seconds.")
    print(f"Train processed shape: {df_train_processed.shape}")
    
    # 4. Transform the Validation Set
    print("Transforming validation set...")
    start_time = time.time()
    
    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col]
    
    X_val_processed = pipeline.transform(X_val)
    
    # Re-attach target column matching validation index alignment
    df_val_processed = X_val_processed.copy()
    df_val_processed[target_col] = y_val.loc[X_val_processed.index]
    
    print(f"Validation preprocessing complete in {time.time() - start_time:.2f} seconds.")
    print(f"Validation processed shape: {df_val_processed.shape}")
    
    # 5. Save the output processed datasets
    print("Saving processed datasets...")
    train_out_path = output_dir / f"{pipeline_name}_train.csv"
    val_out_path = output_dir / f"{pipeline_name}_val.csv"
    
    df_train_processed.to_csv(train_out_path, index=False)
    df_val_processed.to_csv(val_out_path, index=False)
    
    print(f"Saved processed train to: {train_out_path}")
    print(f"Saved processed validation to: {val_out_path}")

def main():
    """Main execution block loading raw data and running the pipelines."""
    # Setup directories
    raw_data_path = Path("data/raw/application_train.csv")
    output_dir = Path("data/processed")
    reports_dir = Path("reports")
    configs_dir = Path("configs/preprocessing")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    target_col = "TARGET"
    
    print("Loading raw dataset...")
    df = pd.read_csv(raw_data_path)
    print(f"Loaded raw dataset with shape: {df.shape}")
    
    # Splitting into train and validation.
    # Why stratified: The credit default label is highly imbalanced (~8% default rate). 
    # Stratified splits guarantee that both train and validation splits contain 
    # the exact same label ratio, preventing metric skew.
    print("Splitting dataset into train/validation (80/20)...")
    df_train, df_val = train_test_split(df, test_size=0.20, random_state=42, stratify=df[target_col])
    print(f"Train set shape: {df_train.shape}")
    print(f"Validation set shape: {df_val.shape}")
    
    # Run pipelines for the three target architectures
    configs = [
        configs_dir / "tree_based.yaml",
        configs_dir / "linear.yaml",
        configs_dir / "neural_network.yaml"
    ]
    
    for config_path in configs:
        if config_path.exists():
            run_pipeline_for_config(config_path, df_train, df_val, output_dir, reports_dir, target_col)
        else:
            print(f"Config not found: {config_path}")
            
    print("\nAll preprocessing pipelines executed successfully!")

if __name__ == "__main__":
    main()
