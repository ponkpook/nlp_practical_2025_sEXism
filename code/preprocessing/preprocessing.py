from typing import Callable, List
import pandas as pd

class DataFramePreprocessor:
    def __init__(self, steps: List[Callable[[pd.Series], pd.Series]]):
        """
        steps: a list of functions, each taking and returning a pd.Series
        """
        self.steps = steps

    def __call__(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df = df.copy()
        series = df[column]
        for fn in self.steps:
            series = fn(series)
        df[column] = series
        return df