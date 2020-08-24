import json
import requests
from app import app, db
import pandas as pd
from app.models import User, Patient, CremeModel


def years_to_days(years):
    return years * 365


def calculate_bmi(weight, height):
    return float(weight) / ((height / 100) ** 2)


def diagnose(patient):
    input_data = {
        'age': years_to_days(int(patient.age)),
        'gender': int(patient.gender),
        'height': int(patient.height),
        'weight': float(patient.weight),
        'ap_hi': int(patient.ap_hi),
        'ap_lo': int(patient.ap_lo),
        'cholesterol': int(patient.cholesterol),
        'gluc': int(patient.gluc),
        'smoke': int(patient.smoke),
        'alco': int(patient.alco),
        'active': int(patient.active),
        'bmi': float(patient.bmi)
    }
    model = CremeModel.query.filter_by(name="BestModel").first()
    prediction = model.pipeline.predict_one(input_data)

    return prediction
