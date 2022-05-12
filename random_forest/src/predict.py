import joblib
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from fastapi import FastAPI, Depends
import pandas as pd
import sys
from . import config
import logging
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

stream_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(f"{config.BASE_DIR}/api.log")

stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


class PredictionInput(BaseModel):
    pclass: int
    name: str
    sex: str
    age: str
    sibsp: int
    parch: int
    ticket: str
    fare: str
    cabin: str
    embarked: str
    boat: str
    body: str


class PredictionOutput(BaseModel):
    prediction: int


class TitanicModel:
    prod_model: Pipeline

    def load_model(self):
        """Loads the model"""
        self.prod_model = joblib.load(config.MODEL_NAME)

    def predict(self, input: PredictionInput) -> PredictionOutput:
        """Runs a prediction"""
        # LOG Aqui INFO
        df = pd.DataFrame([input.dict()])
        if not self.prod_model:
            raise RuntimeError("Model is not loaded")
        prediction = self.prod_model.predict(df)[0]
        results = {"input_raw": input.dict(), "prediction": str(prediction)}
        logger.info(f"Prediction:{json.dumps(results)}")
        print(f"{config.BASE_DIR}/api.log")
        return PredictionOutput(prediction=prediction)


app = FastAPI()
titanic_model = TitanicModel()


@app.post("/prediction")
async def prediction(
    output: PredictionOutput = Depends(titanic_model.predict),
) -> PredictionOutput:
    return output


@app.post("/")
async def root():
    return "Hello World"


@app.on_event("startup")
async def startup():
    # Possible Log: Try and Except
    titanic_model.load_model()
