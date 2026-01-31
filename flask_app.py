from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Placement Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([[
        data["IQ"],
        data["CGPA"],
        data["Tenth"],
        data["Twelfth"],
        data["Comm_Skill"],
        data["Tech_Skills"],
        data["Comm"],
        data["Hackathons"],
        data["Certifications"],
        data["Backlogs"]
    ]])

    prediction = int(model.predict(features)[0])
    confidence = max(model.predict_proba(features)[0]) * 100

    return jsonify({
        "Placed": prediction,
        "Confidence (%)": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
