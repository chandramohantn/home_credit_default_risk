import pytest
import pandas as pd
import numpy as np
from src.preprocessing.transformers.cleaning import DataCleaner
from src.preprocessing.transformers.missing import MissingValueTransformer
from src.preprocessing.transformers.rare_categories import RareCategoryTransformer
from src.preprocessing.transformers.encoding import EncodingTransformer
from src.preprocessing.transformers.outliers import OutlierTransformer
from src.preprocessing.transformers.transformations import FeatureTransformer
from src.preprocessing.transformers.scaling import FeatureScaler
from src.preprocessing.pipeline import PreprocessingPipeline

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "ID": range(n),
        "income": np.random.exponential(scale=50000, size=n),
        "age": np.random.randint(20, 70, size=n).astype(float),
        "gender": np.random.choice(["M", "F", None], size=n, p=[0.45, 0.45, 0.10]),
        "occupation": np.random.choice(["Engineer", "Doctor", "Teacher", "Astronaut"], size=n, p=[0.40, 0.30, 0.28, 0.02]),
        "target": np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    })
    # Add some nulls to numeric columns
    df.loc[df.sample(10).index, "income"] = np.nan
    df.loc[df.sample(5).index, "age"] = np.nan
    df.loc[df.sample(10).index, "gender"] = np.nan
    
    # Add an outlier to income
    df.loc[5, "income"] = 5000000.0
    return df

def test_data_cleaner(sample_data):
    # Add duplicate row and duplicate column
    df = sample_data.copy()
    duplicate_row = df.iloc[[0]].copy()
    df = pd.concat([df, duplicate_row], ignore_index=True)
    df["income_duplicate"] = df["income"]

    cleaner = DataCleaner(drop_row_duplicates=True, drop_col_duplicates=True)
    
    # Fit & Transform
    df_clean = cleaner.fit_transform(df)
    
    # Assert row duplicates were removed (len goes from 101 back to 100)
    assert len(df_clean) == 100
    
    # Assert column duplicates were removed (income_duplicate is dropped)
    assert "income_duplicate" not in df_clean.columns
    assert "income" in df_clean.columns

def test_missing_value_transformer(sample_data):
    df = sample_data.copy()
    
    # Fit imputer
    imputer = MissingValueTransformer(
        num_strategy="median",
        cat_strategy="constant",
        cat_fill_value="MissingOcc",
        add_indicators=True
    )
    df_imputed = imputer.fit_transform(df)
    
    # Assert no nulls remaining
    assert df_imputed["income"].isnull().sum() == 0
    assert df_imputed["age"].isnull().sum() == 0
    assert df_imputed["gender"].isnull().sum() == 0
    
    # Assert missing indicators were added
    assert "income_isnull" in df_imputed.columns
    assert "age_isnull" in df_imputed.columns
    
    # Check categorical missing imputation
    assert "MissingOcc" in df_imputed["gender"].values

def test_rare_category_transformer(sample_data):
    df = sample_data.copy()
    # Fill nulls first
    df["occupation"] = df["occupation"].fillna("Teacher")
    
    # Fit rare category transformer with high threshold pct to trigger "Astronaut" grouping
    rare_handler = RareCategoryTransformer(threshold_pct=0.05, fill_value="RARE")
    df_rare = rare_handler.fit_transform(df)
    
    # Astronaut should be mapped to RARE
    astronauts_original = df["occupation"] == "Astronaut"
    assert (df_rare.loc[astronauts_original, "occupation"] == "RARE").all()

def test_encoding_transformer(sample_data):
    df = sample_data.copy()
    # Impute missing values first to encode safely
    df["gender"] = df["gender"].fillna("Unknown")
    
    encoder = EncodingTransformer(
        default_strategy="target",
        column_strategies={"occupation": "one_hot"},
        target_cv_folds=3
    )
    
    df_encoded = encoder.fit_transform(df, df["target"])
    
    # Assert occupation is one-hot encoded
    assert "occupation_Engineer" in df_encoded.columns
    assert "occupation_Teacher" in df_encoded.columns
    
    # Assert gender is target encoded (remains a single numerical column)
    assert "gender" in df_encoded.columns
    assert df_encoded["gender"].dtype in [np.float64, float]

def test_catboost_encoding_transformer(sample_data):
    df = sample_data.copy()
    df["gender"] = df["gender"].fillna("Unknown")

    encoder = EncodingTransformer(
        default_strategy="catboost",
        target_cv_folds=3,
        catboost_prior=5.0,
    )

    df_encoded = encoder.fit_transform(df[["gender", "occupation"]], df["target"])

    assert "gender" in df_encoded.columns
    assert "occupation" in df_encoded.columns
    assert df_encoded["gender"].dtype in [np.float64, float]

def test_outlier_transformer(sample_data):
    df = sample_data.copy()
    df["income"] = df["income"].fillna(df["income"].median())
    
    outliers = OutlierTransformer(strategy="clip", method="iqr", threshold=1.5)
    df_treated = outliers.fit_transform(df)
    
    # Assert upper value was capped
    original_max = df["income"].max()
    new_max = df_treated["income"].max()
    assert new_max < original_max
    assert new_max == pytest.approx(outliers.limits_["income"][1])

def test_feature_scaler(sample_data):
    df = sample_data.copy()
    df["income"] = df["income"].fillna(df["income"].median())
    df["age"] = df["age"].fillna(df["age"].median())
    
    scaler = FeatureScaler(default_strategy="standard")
    df_scaled = scaler.fit_transform(df[["income", "age"]])
    
    # Assert mean is close to 0 and std is close to 1
    assert abs(df_scaled["income"].mean()) < 1e-7
    assert abs(df_scaled["income"].std() - 1.0) < 0.1  # sample size 100 might be slightly different standard deviation

def test_pipeline_integration(sample_data, tmp_path):
    df = sample_data.copy()
    y = df.pop("target")
    
    config = {
        "pipeline": {
            "schema_validation": {
                "enabled": True,
                "strict": True,
            },
            "leakage_validation": {
                "enabled": True,
                "fail_on_detection": False,
            },
            "metadata": {
                "version": "2.0",
            },
            "available_components": [
                "cleaner",
                "missing",
                "rare_categories",
                "encoder",
                "outliers",
                "transformations",
                "scaler",
            ],
            "steps": [
                {
                    "component": "cleaner",
                    "params": {
                        "drop_row_duplicates": True,
                        "drop_col_duplicates": True,
                        "coerce_objects_to_category": True,
                    },
                },
                {
                    "component": "missing",
                    "params": {
                        "num_strategy": "median",
                        "cat_strategy": "constant",
                        "cat_fill_value": "Unknown",
                        "add_indicators": True,
                    },
                },
                {
                    "component": "rare_categories",
                    "params": {
                        "threshold_pct": 0.05,
                        "fill_value": "RARE",
                    },
                },
                {
                    "component": "encoder",
                    "params": {
                        "default_strategy": "catboost",
                        "column_strategies": {"occupation": "one_hot"},
                        "target_cv_folds": 3,
                    },
                },
                {
                    "component": "outliers",
                    "params": {
                        "strategy": "clip",
                        "method": "iqr",
                    },
                },
                {
                    "component": "transformations",
                    "params": {
                        "transformations": {
                            "income": "log1p",
                        },
                    },
                },
                {
                    "component": "scaler",
                    "params": {
                        "default_strategy": "standard",
                    },
                },
            ],
        }
    }
    
    pipeline = PreprocessingPipeline(config, reports_dir=str(tmp_path), target_column="target")
    
    # Run fit_transform
    df_processed = pipeline.fit_transform(df, y)
    
    # Check reports were written
    assert (tmp_path / "preprocessing_report.json").exists()
    assert (tmp_path / "missing_value_report.json").exists()
    assert (tmp_path / "outlier_report.json").exists()
    assert (tmp_path / "encoding_report.json").exists()
    assert (tmp_path / "scaling_report.json").exists()
    assert (tmp_path / "transformation_report.json").exists()
    assert (tmp_path / "feature_metadata.json").exists()
    
    # Run transform
    df_test = pipeline.transform(df)
    
    # Assert shapes are equal
    assert df_processed.shape[1] == df_test.shape[1]

def test_strict_schema_validation_rejects_missing_columns(sample_data, tmp_path):
    df = sample_data.copy()
    y = df.pop("target")

    config = {
        "pipeline": {
            "schema_validation": {
                "enabled": True,
                "strict": True,
            },
            "steps": [
                {
                    "component": "missing",
                    "params": {
                        "num_strategy": "median",
                        "cat_strategy": "constant",
                        "cat_fill_value": "Unknown",
                    },
                },
            ],
        }
    }

    pipeline = PreprocessingPipeline(config, reports_dir=str(tmp_path), target_column="target")
    pipeline.fit_transform(df, y)

    with pytest.raises(ValueError):
        pipeline.transform(df.drop(columns=["income"]))
