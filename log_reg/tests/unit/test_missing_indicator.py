from sre_constants import CATEGORY_UNI_LINEBREAK

import numpy as np
import pandas as pd
import pytest
from titanic_pipeline.training.transformers import MissingIndicator


def obtener_datos_missin_indicator():
    """En esta función se obtienen los datos para el test de MissingIndicator

    Returns:
        List[Tuples(longitud, DataFrame, List[columnas])]: Regresa una
        lista de tuplas con los datos para el test de MissingIndicator|
    """
    titles = ["Mrs", np.nan, "Mr", np.nan, "Miss", "Master", "Other"]
    titles_variables = ["title"]
    # la longitud es más 1 columna, porque el MissingIndicator agrega una columna
    titles_len = len(titles_variables) + 1
    titles_df = pd.DataFrame(titles, columns=titles_variables)

    # aquí no agregué np.nan pero de todas formas debe agregar la columna el MissingIndicator
    cabins = ["B5", "E12", "D7", "A36", "C101", "C62", "B35", "A23"]
    cabins_variables = ["cabin"]
    cabins_len = len(cabins_variables) + 1
    cabins_df = pd.DataFrame(cabins, columns=cabins_variables)

    return [
        (titles_len, titles_df, titles_variables),
        (cabins_len, cabins_df, cabins_variables),
    ]


@pytest.mark.parametrize("result, df, variables", obtener_datos_missin_indicator())
def test_missing_indicator(result, df, variables):
    """missing_indicator: Test para verificar que el MissingIndicator
    funciona correctamente

    Args:
        result (Int): cantidad de columnas que debe tener el DataFrame
        df (DataFrame): DataFrame con los datos para el test
        variables (List[Str]): lista de variables con la(s) columna(s) de target
    """
    # traer el MissingIndicator y hacer las pruebas, debe pasar de 1 columna a 2
    missing = MissingIndicator(variables)
    # missing.fit(df)
    df = missing.transform(df)
    print(f"df.shape = {df.shape}")
    print(f"df.columns = {df.columns}")
    assert len(df.columns) == result
