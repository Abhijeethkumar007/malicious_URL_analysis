from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import requests
import csv
from features.xgbmodel_features import get_prediction_from_url
from features.rfmodel_features import get_prediction_from_url_rf

app = Flask(__name__)

# Load GBM and RF models
with open('models/xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)
with open('models/random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

def expand_url(url):
    try:
        response = requests.head(url, allow_redirects=True)
        return response.url
    except requests.RequestException:
        return url

def log_to_csv(url, expanded_url, xgb_prediction, rf_prediction, prediction):
    with open('logs/url_predictions.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([url, expanded_url, xgb_prediction[0], rf_prediction[0], prediction])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    
    # Expand the URL if it's a short URL
    expanded_url = expand_url(url)

    # Pass the expanded URL to xgbmodel_features.py
    features1 = get_prediction_from_url(expanded_url)
    # Pass the expanded URL to rfmodel_features.py
    features2 = get_prediction_from_url_rf(expanded_url)
    
    # Due to updates to scikit-learn, we now need a 2D array as a parameter to the predict function.
    features1 = np.array(features1).reshape((1, -1))
    
    # Make predictions using GBM model
    xgb_prediction = xgb_model.predict(features1)
    # Make predictions using RF model
    rf_prediction = rf_model.predict(features2)
    
    # Combine predictions
    combined_prediction = (xgb_prediction + rf_prediction[0]) / 2

    # Determine the final prediction
    if rf_prediction == 1 and xgb_prediction >= 2:  # Adjust the threshold as per your needs
        prediction = 'Malicious'
    elif rf_prediction == 1 and xgb_prediction == 2 or xgb_prediction == 1:
        prediction = 'Unsafe'
    else:
        prediction = 'Safe'
    
    # Log the URL and prediction details to CSV
    log_to_csv(url, expanded_url, xgb_prediction, rf_prediction, prediction)
    
    return render_template('result.html', prediction=prediction, url=expanded_url)

if __name__ == '__main__':
    app.run(debug=True)
