import xgboost as xgb

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
model.save_model('best_model_updated.json')
loaded_model = xgb.Booster()
loaded_model.load_model('best_model_updated.json')