from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

logreg = pickle.load(open("model\logreg.pkl", "rb"))

scaler = pickle.load(open("model\scaler.pkl", "rb"))


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def make_prediction():

    if(request.method == 'POST') : 
        Pregnancies = float(request.form.get('Pregnancies'))

        Glucose = float(request.form.get('Glucose'))

        BloodPressure = float(request.form.get('BloodPressure'))

        SkinThickness = float(request.form.get('SkinThickness'))

        Insulin = float(request.form.get('Insulin'))

        BMI = float(request.form.get('BMI'))

        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))

        Age = float(request.form.get('Age'))

        data  = [ Pregnancies,Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,Age]

        new_scaled_data = scaler.transform([data])

        op = logreg.predict(new_scaled_data)

        if op==1:
            result = "Person has diabetes"
        else:
            result = "Person do not have diabetes"

        return render_template("home.html", result = result)
    else:
        return render_template("home.html")


if __name__ == '__main__' :
    app.run(host = "0.0.0.0")

