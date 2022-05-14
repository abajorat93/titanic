import os
BASE_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
URL = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
from .configs import train_config
MODEL_NAME = os.path.realpath(os.path.join(BASE_DIR, "models", train_config.MODEL, "model.sav"))

SEED_SPLIT = 404
SEED_MODEL = 404
MODEL_NAME = os.path.realpath(os.path.join(BASE_DIR, "models", "model.sav"))
TARGET = "survived"
FEATURES = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "cabin",
    "embarked",
    "title",
]
NUMERICAL_VARS = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_VARS = ["sex", "cabin", "embarked", "title"]
DROP_COLS = ["boat", "body", "ticket", "name"]
