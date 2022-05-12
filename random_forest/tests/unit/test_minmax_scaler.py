import pytest
import pandas as pd
from titanic_pipeline.training.transformers import MinMaxScaler
import numpy as np


def GetMinMaxScaleTestData():
    df = pd.DataFrame([[1, 1], [2, 0], [3, 1], [4, 0], [5, 1]], columns=["a", "b"])
    result = pd.DataFrame(
        [[0, 1.0], [0.25, 0.0], [0.5, 1.0], [0.75, 0.0], [1, 1.0]], columns=["a", "b"]
    )
    return [(df, result)]


@pytest.mark.parametrize("df, result", GetMinMaxScaleTestData())
def test_extract_only_letter(df, result):
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(df)
    test_result = minmax_scaler.transform(df)
    assert pd.DataFrame(test_result, columns=["a", "b"]).equals(result)
