import pandas as pd
import numpy as np
import re
import pytest


def ExtractOnlyLetter(x):
    if type(x) == str:
        return "".join(re.findall("[a-zA-Z]+", x))
    return x


def ExtractOnlyLetterResult():
    return [
        (
            pd.DataFrame(["A1", "B1", "C1", "CC2", "$D1", np.nan], columns=["cabin"]),
            pd.DataFrame(["A", "B", "C", "CC", "D", np.nan], columns=["cabin"]),
        )
    ]


@pytest.mark.parametrize("df, result", ExtractOnlyLetterResult())
def test_extract_only_letter(df, result):
    test_result = df["cabin"].apply(ExtractOnlyLetter).to_frame()
    assert test_result.equals(result)
