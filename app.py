from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np



# -------------------------
# Define Neural Network
# -------------------------
class Net(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(128, output_dim)  # logits
        )

    def forward(self, x):
        return self.model(x)


# -------------------------
# Load Models and Scaler
# -------------------------
scaler = joblib.load("scaler2.pkl")
xgb_model = joblib.load("xgboost_cpu_model.pkl")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
nn_model = Net(input_dim=15)
nn_model.load_state_dict(torch.load("nn_model2.pth", weights_only=True, map_location=device))
nn_model.to(device)
nn_model.eval()

# Feature order (must match training)
features = [
    "Time_To_Live", "Rate", "psh_flag_number", "ack_count", "syn_count", "rst_count",
    "DNS", "TCP", "UDP", "ARP", "ICMP", "IPv", "Std", "Tot size", "IAT"
]

class_mapping = {0: "Benign", 1: "DoS", 2: "DDoS"}

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)

CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "API is running! Try POST /predict"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.json

        # Ensure all required features are present
        for f in features:
            if f not in data:
                return jsonify({"error": f"Missing feature: {f}"}), 400

        # Convert to DataFrame
        df = pd.DataFrame([data], columns=features)

        # Scale input
        X_scaled = scaler.transform(df)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        # NN prediction
        with torch.no_grad():
            outputs = nn_model(X_tensor).cpu().numpy()
            nn_pred = int(np.argmax(outputs, axis=1)[0])

        # XGBoost prediction
        xgb_pred = int(xgb_model.predict(X_scaled)[0])

        # Hard-coded ensemble logic
        if xgb_pred == 0:
            final_pred = 0
        elif xgb_pred == nn_pred:
            final_pred = xgb_pred
        else:
            final_pred = nn_pred
        # print(class_mapping[final_pred])
        return jsonify({
            "nn_prediction": class_mapping[nn_pred],
            "xgb_prediction": class_mapping[xgb_pred],
            "final_prediction": class_mapping[final_pred]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
