# import sys
# sys.path.insert(0, "../../features")
# sys.path.insert(1, "/c/Tutoriales/Titanic_Pipeline_MLOps_Eq3/src/features")
# sys.path.append("/c/Tutoriales/Titanic_Pipeline_MLOps_Eq3/src/features")
# sys.path.append("features")

from sre_constants import CATEGORY_UNI_LINEBREAK
import pandas as pd
import pytest
from titanic_pipeline.training.transformers import OneHotEncoder


def obtener_datos_one_hot_encoder():
    """En esta función se obtienen los datos para el test de OneHotEncoder

    Returns:
        List[Tuples(longitud, DataFrame, List[columnas])]: Regresa una
        lista de tuplas con los datos para el test de OneHotEncoder
    """
    titles = ["Mrs", "Mr", "Miss", "Master", "Other"]
    # la longitud es menos 1 columna, porque el OneHotEncodig no toma la columna de target
    titles_len = len(titles) - 1
    titles_variables = ["title"]
    titles_df = pd.DataFrame(titles, columns=titles_variables)

    cabins = ["B5", "C22", "E12", "D7", "A36", "C101", "C62", "B35", "A23"]
    cabins_variables = ["cabin"]
    cabins_len = len(cabins) - 1
    cabins_df = pd.DataFrame(cabins, columns=cabins_variables)

    return [
        (titles_len, titles_df, titles_variables),
        (cabins_len, cabins_df, cabins_variables),
    ]


@pytest.mark.parametrize("result, df, variables", obtener_datos_one_hot_encoder())
def test_one_hot_encoding(result, df, variables):
    """one_hot_encoding: Test para verificar que el OneHotEncoder
    funciona correctamente

    Args:
        result (Int): cantidad de columnas que debe tener el DataFrame
        df (DataFrame): DataFrame con los datos para el test
        variables (List[Str]): lista de variables con la(s) columna(s) de target
    """
    # traer el OneHotEncoding de Cesar y hacer las pruebas, debe pasar de 1 columna a 5
    one_hot = OneHotEncoder(variables)
    one_hot.fit(df)
    df = one_hot.transform(df)
    print(f"df.shape = {df.shape}")
    print(f"df.columns = {df.columns}")
    assert len(df.columns) == result
