from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the model and encoders
try:
    model_path = os.path.join('models', 'model.pkl')
    encoder_path = os.path.join('models', 'encoders.pkl')
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
    else:
        model = None
        encoders = None
except Exception as e:
    print(f"Error loading model or encoders: {e}")
    model = None
    encoders = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or encoders is None:
            return jsonify({'error': 'Model or encoders not loaded. Please ensure model is properly trained and saved.'})
        
        # Get form data
        features = {
            'age': float(request.form['age']),
            'job': request.form['job'],
            'marital': request.form['marital'],
            'education': request.form['education'],
            'balance': float(request.form['balance']),
            'housing': request.form['housing'],
            'loan': request.form['loan'],
            'duration': float(request.form['duration']),
            'campaign': float(request.form['campaign']),
            'previous': float(request.form['previous']),
            'month': request.form['month']
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Encode categorical variables
        categorical_columns = ['job', 'marital', 'education', 'housing', 'loan', 'month']
        for col in categorical_columns:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        result = {
            'prediction': 'Yes' if prediction == 1 else 'No',
            'probability': f"{probability:.2%}"
        }
            
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5001)