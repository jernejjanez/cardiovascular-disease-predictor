import json
import requests
from app import app


def feedback(feedback_value):
    if feedback_value == '0':
        return 'Patient does not have the disease'
    elif feedback_value == '1':
        return 'Patient has the disease'
    else:
        return 'None'
