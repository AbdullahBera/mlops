from fastapi import FastAPI
from joblib import load

app = FastAPI()

model_path = "/Users/bera/Desktop/MSDS/603/mlops/labs/reddit_model_pipeline.joblib"
model = load(model_path)

@app.post("/predict")
async def predict(data: dict):
    try:
        text = data['text']
        prediction = model.predict([text])[0]
        # Add probability scores
        probabilities = model.predict_proba([text])[0]
        
        return {
            "prediction": int(prediction),
            "probabilities": {
                "class_0": float(probabilities[0]),
                "class_1": float(probabilities[1])
            }
        }
    except Exception as e:
        return {"error": str(e)}