from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("category_model.pkl")

@app.route("/")
def home():
    return "Jago Category API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    subject = data.get("subject", "")
    counterparty = data.get("counterparty", "")
    direction = data.get("direction", "")

    text = f"{subject} {counterparty} {direction}"

    prediction = model.predict([text])[0]

    return jsonify({
        "category": prediction
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)