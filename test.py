'''
This file initiates a test to determine whether the pickle file can accurately predict the output based on the provided features.
'''

import pickle
import pandas as pd

test_data = {
    "CUSTOMER_ID": 2924,
    "TERMINAL_ID": 345678,
    "TX_AMOUNT": 80.0,
    "TX_TIME_SECONDS": 333233,
    "TX_TIME_DAYS": 33,
    "TX_YEAR": 2018,
    "TX_MONTH": 5,
    "TX_NIGHT": 0,
    "TX_WEEKEND": 1,
    "TX_IS_AMOUNT_HIGH": 0,
    "TX_TIME_TAKEN_HIGH": 0,
    "FRAUD_SCENARIO_Large_Amount": 0,
    "FRAUD_SCENARIO_Leaked_data": 0,
    "FRAUD_SCENARIO_Legitimate": 1,
    "FRAUD_SCENARIO_Random_Fraud": 0,
    "TX_HOUR": 4,
    "TX_RUSH_HOUR": 0
}

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.DataFrame([list(test_data.values())], columns=list(test_data.keys()))

prediction = model.predict(df)

print("Prediction:", prediction)
print("Prediction Type:", type(prediction))
