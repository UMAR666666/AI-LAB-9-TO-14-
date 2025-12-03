from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import sys

app = Flask(__name__)

# Load the model and scaler
model_path = r"C:\Users\HAROON-CHISHTI\Desktop\Project\diabetes_model.pkl"
scaler_path = r"C:\Users\HAROON-CHISHTI\Desktop\Project\scaler.pkl"
feature_names_path = r"C:\Users\HAROON-CHISHTI\Desktop\Project\feature_names.pkl"

model = None
scaler = None
feature_names = None

# Try to load model and scaler
try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print("✓ Model loaded successfully")
    else:
        print(f"⚠ Model file not found at {model_path}")
        print("  Please run: python ../train_model.py")
except Exception as e:
    print(f"✗ Error loading model: {e}")

try:
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        print("✓ Scaler loaded successfully")
    else:
        print(f"⚠ Scaler file not found at {scaler_path}")
except Exception as e:
    print(f"✗ Error loading scaler: {e}")

try:
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'rb') as file:
            feature_names = pickle.load(file)
        print("✓ Feature names loaded successfully")
    else:
        print(f"⚠ Feature names file not found at {feature_names_path}")
except Exception as e:
    print(f"✗ Error loading feature names: {e}")

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    # Check if model and scaler are loaded
    if model is None or scaler is None:
        return jsonify({
            'error': 'Model not loaded. Please run: python ../train_model.py first'
        }), 503
    
    try:
        data = request.json
        
        # Extract features from the request
        features = {
            'gender_Male': 1 if data.get('gender') == 'Male' else 0,
            'gender_Other': 1 if data.get('gender') == 'Other' else 0,
            'age': float(data.get('age', 0)),
            'hypertension': int(data.get('hypertension', 0)),
            'heart_disease': int(data.get('heart_disease', 0)),
            'smoking_history_current': 1 if data.get('smoking_history') == 'current' else 0,
            'smoking_history_ever': 1 if data.get('smoking_history') == 'ever' else 0,
            'smoking_history_former': 1 if data.get('smoking_history') == 'former' else 0,
            'smoking_history_never': 1 if data.get('smoking_history') == 'never' else 0,
            'smoking_history_not current': 1 if data.get('smoking_history') == 'not_current' else 0,
            'bmi': float(data.get('bmi', 0)),
            'HbA1c_level': float(data.get('HbA1c_level', 0)),
            'blood_glucose_level': float(data.get('blood_glucose_level', 0))
        }
        
        # Convert to DataFrame with correct feature order
        if feature_names:
            df = pd.DataFrame([features])[feature_names]
        else:
            df = pd.DataFrame([features])
        
        # Scale the features
        features_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        result = {
            'prediction': int(prediction),
            'prediction_text': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability': {
                'non_diabetic': float(probability[0]),
                'diabetic': float(probability[1])
            },
            'confidence': float(max(probability)) * 100
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
