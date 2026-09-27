from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
import numpy as np

model = {}
encoders = {}
feature_names = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global feature_names

    # Load the CatBoost model
    model["Catmodel"] = joblib.load("CatBoost.pkl")
    model["catThresold"] = joblib.load("catThresold.pkl")

    # Get feature names from the model
    try:
        feature_names = model["Catmodel"].feature_names_
    except AttributeError:
        # Fallback feature names in correct order
        feature_names = [
            "person_age",
            "person_income",
            "person_home_ownership",
            "person_emp_length",
            "loan_intent",
            "loan_grade",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_default_on_file",
            "cb_person_cred_hist_length",
        ]

    yield

    model.clear()
    encoders.clear()


app = FastAPI(lifespan=lifespan)


class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


def preprocess_data(data: LoanApplication) -> pd.DataFrame:
    """
    Preprocess the input data to match what the model expects
    """
    # Create DataFrame from input data
    df = pd.DataFrame([data.model_dump()])

    # Ensure all columns exist and are in the right order
    df = df[
        [
            "person_age",
            "person_income",
            "person_home_ownership",
            "person_emp_length",
            "loan_intent",
            "loan_grade",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_default_on_file",
            "cb_person_cred_hist_length",
        ]
    ]

    # Convert data types
    df["person_age"] = df["person_age"].astype(int)
    df["person_income"] = df["person_income"].astype(float)
    df["person_emp_length"] = df["person_emp_length"].astype(float)
    df["loan_amnt"] = df["loan_amnt"].astype(float)
    df["loan_int_rate"] = df["loan_int_rate"].astype(float)
    df["loan_percent_income"] = df["loan_percent_income"].astype(float)
    df["cb_person_cred_hist_length"] = df["cb_person_cred_hist_length"].astype(int)

    # Ensure categorical columns are strings
    df["person_home_ownership"] = df["person_home_ownership"].astype(str)
    df["loan_intent"] = df["loan_intent"].astype(str)
    df["loan_grade"] = df["loan_grade"].astype(str)
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].astype(str)

    return df


@app.post("/predict")
def predict(data: LoanApplication):
    try:
        # Preprocess input data
        input_df = preprocess_data(data)

        # Make prediction
        probability = model["Catmodel"].predict_proba(input_df)[:, 1][0]
        threshold = float(model["catThresold"])

        # Determine prediction
        prediction = int(probability >= threshold)

        # Calculate confidence (distance from threshold)
        distance_from_threshold = abs(probability - threshold)
        max_distance = max(threshold, 1 - threshold)
        confidence = min((distance_from_threshold / max_distance) * 100, 100)

        return {
            "default_probability": float(probability),
            "default_prediction": prediction,
            "thresold": threshold,
            "result": "High Risk" if prediction == 1 else "Low Risk",
            "confidence": float(confidence),
        }

    except Exception as e:
        # Return detailed error message for debugging
        return {
            "error": str(e),
            "type": type(e).__name__,
            "message": f"Error during prediction: {str(e)}",
        }


app.mount("/", StaticFiles(directory="static", html=True), name="static")