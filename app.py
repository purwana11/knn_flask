# app.py

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model = joblib.load("model_knn.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Ambil input user dari form
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            # Buat array dan normalisasi
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            features_scaled = scaler.transform(features)

            # Prediksi dengan model KNN
            prediction = model.predict(features_scaled)[0]
        except ValueError:
            prediction = "Input tidak valid. Harap masukkan angka yang benar."
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
