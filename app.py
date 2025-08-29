from flask import Flask
from flask_cors import CORS
import joblib

app = Flask(__name__)

CORS(app)

scaler = joblib.load("scaler2.pkl")
xgb_model = joblib.load("xgboost_cpu_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return "API is running! Try POST /predict"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)