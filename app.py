from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    size = request.form.get('size')
    bedrooms = request.form.get('bedrooms')
    location = request.form.get('location')

    # Preprocess input
    input_data = pd.DataFrame([[size, bedrooms, location]], columns=['size', 'bedrooms', 'location'])
    input_data = pd.get_dummies(input_data, columns=['location'], drop_first=True)

    # Ensure the input data has the same columns as the training data
    required_columns = model.feature_names_in_
    missing_cols = set(required_columns) - set(input_data.columns)
    for c in missing_cols:
        input_data[c] = 0

    input_data = input_data[required_columns]
    
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
