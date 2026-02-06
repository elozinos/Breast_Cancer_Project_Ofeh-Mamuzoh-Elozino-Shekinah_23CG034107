from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained Logistic Regression model
model_path = os.path.join("model", "logistic_regression_breast_cancer_model.pkl")
model = joblib.load(model_path)

# Home page
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Collect input values from the form
            input_data = {
                "mean perimeter": [float(request.form["mean_perimeter"])],
                "mean area": [float(request.form["mean_area"])],
                "mean concavity": [float(request.form["mean_concavity"])],
                "mean radius": [float(request.form["mean_radius"])],
                "mean compactness": [float(request.form["mean_compactness"])]
            }

            df = pd.DataFrame(input_data)

            # Make prediction
            pred = model.predict(df)[0]
            prediction = "Benign" if pred == 1 else "Malignant"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
