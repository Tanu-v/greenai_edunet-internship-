from flask import Flask, request, jsonify
import pandas as pd
import joblib
import config

# Load saved model & scaler
model = joblib.load(config.MODEL_PATH)
scaler = joblib.load(config.SCALER_PATH)

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "AQI Monitoring API is running 🚀"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])

        # Scale features
        features = scaler.transform(df)

        # Predict
        prediction = model.predict(features)[0]
        return jsonify({"AQI_Bucket_Predicted": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=8080)
