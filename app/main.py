'''from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
app = FastAPI()

class predict(BaseModel):
    CUSTOMER_ID: int
    TERMINAL_ID: int
    TX_AMOUNT: float
    TX_TIME_SECONDS: int
    TX_TIME_DAYS: int  # Appropriate value
    TX_YEAR: int
    TX_MONTH: int
    TX_NIGHT: int  # 0 or 1
    TX_WEEKEND: int  # 0 or 1
    TX_IS_AMOUNT_HIGH: int  # 0 or 1
    TX_TIME_TAKEN_HIGH: int  # 0 or 1
    FRAUD_SCENARIO_Large_Amount: int  # 0 or 1
    FRAUD_SCENARIO_Leaked_data: int  # 0 or 1
    FRAUD_SCENARIO_Legitimate: int  # 0 or 1
    FRAUD_SCENARIO_Random_Fraud: int  # 0 or 1
    TX_HOUR: int
    TX_RUSH_HOUR: int  # 0 or 1

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/")
async def prediction(item: predict):
    df = pd.DataFrame([item.dict().values()], columns = item.dict().keys())
    yhat = model.predict(df)
    return {"prediction": int(yhat)}
'''
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd

app = FastAPI()

class PredictModel(BaseModel):
    CUSTOMER_ID: int = Field(...)
    TERMINAL_ID: int = Field(...)
    TX_AMOUNT: float = Field(...)
    TX_TIME_SECONDS: int = Field(...)
    TX_TIME_DAYS: int = Field(...)
    TX_YEAR: int = Field(...)
    TX_MONTH: int = Field(...)
    TX_NIGHT: int = Field(ge=0, le=1)
    TX_WEEKEND: int = Field(ge=0, le=1)
    TX_IS_AMOUNT_HIGH: int = Field(ge=0, le=1)
    TX_TIME_TAKEN_HIGH: int = Field(ge=0, le=1)
    FRAUD_SCENARIO_Large_Amount: int = Field(ge=0, le=1)
    FRAUD_SCENARIO_Leaked_data: int = Field(ge=0, le=1)
    FRAUD_SCENARIO_Legitimate: int = Field(ge=0, le=1)
    FRAUD_SCENARIO_Random_Fraud: int = Field(ge=0, le=1)
    TX_HOUR: int = Field(...)
    TX_RUSH_HOUR: int = Field(ge=0, le=1)

@app.post("/")
async def prediction(item: PredictModel):
    df = pd.DataFrame([item.model_dump().values()], columns=item.model_dump().keys())
    yhat = model.predict(df)
    return {"prediction": int(yhat[0])}

# Load model outside of route
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)