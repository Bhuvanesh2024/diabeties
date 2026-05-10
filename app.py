from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("standard_scaler.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features")

    if not features or len(features) != 8:
        return jsonify({"error": "Provide 'features' as a list of 8 numeric values"}), 400

    df = pd.DataFrame([features], columns=scaler.feature_names_in_)
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0].tolist()

    return jsonify({
        "prediction": int(prediction),
        "probability": {"class_0": probability[0], "class_1": probability[1]}
    })

if __name__ == "__main__":
    app.run(debug=True)
