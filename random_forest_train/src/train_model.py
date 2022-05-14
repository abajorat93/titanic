import joblib
from transformers.transformers import (
    MissingIndicator,
    CabinOnlyLetter,
    CategoricalImputerEncoder,
    NumericalImputesEncoder,
    RareLabelCategoricalEncoder,
    OneHotEncoder,
    MinMaxScaler,
    CleaningTransformer,
    DropTransformer,
)
import pandas as pd

from . import config
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import sys


def train(max_depth: int = 4):
    numeric_transformer = Pipeline(
        steps=[
            ("missing_indicator", MissingIndicator(config.NUMERICAL_VARS)),
            ("median_imputation", NumericalImputesEncoder(config.NUMERICAL_VARS)),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("cabin_only_letter", CabinOnlyLetter("cabin")),
            ("categorical_imputer", CategoricalImputerEncoder(config.CATEGORICAL_VARS)),
            (
                "rare_labels",
                RareLabelCategoricalEncoder(
                    tol=0.02, variables=config.CATEGORICAL_VARS
                ),
            ),
            ("one_hot", OneHotEncoder(config.CATEGORICAL_VARS)),
        ]
    )

    preprocessor = Pipeline(
        [
            ("cleaning", CleaningTransformer()),
            ("categorical", categorical_transformer),
            ("numeric", numeric_transformer),
            ("dropper", DropTransformer(config.DROP_COLS)),
            ("scaling", MinMaxScaler()),
        ]
    )

    regressor = RandomForestClassifier(
        max_depth=max_depth, class_weight="balanced", random_state=config.SEED_MODEL
    )

    titanic_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("random_forest_regressor", regressor)]
    )
    df = pd.read_csv(config.URL).drop(columns="home.dest")
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(config.TARGET, axis=1),
        df[config.TARGET],
        test_size=0.2,
        random_state=config.SEED_SPLIT,
    )
    titanic_pipeline.fit(X_train, y_train)
    preds = titanic_pipeline.predict(X_test)
    accuracy = (preds == y_test).sum() / len(y_test)
    print(f"Accuracy of the model is {accuracy}")

    # now = datetime.now()
    # date_time = now.strftime("%Y_%d_%m_%H%M%S")
    filename = f"{config.MODEL_NAME}"
    print(f"Model stored in models as {filename}")
    joblib.dump(titanic_pipeline, f"{config.MODEL_NAME}")


if __name__ == "__main__":
    max_depth = float(sys.argv[1]) if len(sys.argv) > 1 else 4
    train(max_depth)
