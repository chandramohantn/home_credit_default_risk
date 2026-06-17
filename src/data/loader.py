import pandas as pd
from pathlib import Path


class DatasetLoader:

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def load_csv(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(
            self.data_dir / filename
        )