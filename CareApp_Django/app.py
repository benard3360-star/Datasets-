# app.py
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*version.*')

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

# Load Model, Top Features, Encoder, and Scaler
with open("catboost_tuned_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("top_features.pkl", "rb") as f:
    top_features = pickle.load(f)

with open("categorical_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("scaler_stepwise_features.pkl", "rb") as f:
    scaler = pickle.load(f)

# Form categorical features
form_categorical_features = ['Final Grade', 'County', 'Sub_County', 'Location', 'Course']

# Correct encoder mapping based on actual encoder contents
cat_to_encoder_idx = {
    'County': 2,
    'Sub_County': 3,
    'Location': 4,
    'Course': 7,
    'Final Grade': 9
}

# Load counties data
with open("kenyan_counties.json", "r") as f:
    counties_data = json.load(f)

# Create county-subcounty mapping
county_subcounty_map = {county['name']: county['sub_counties'] for county in counties_data}

# Update categorical values to use counties from JSON
categorical_values = {feat: encoders[idx].classes_.tolist() for feat, idx in cat_to_encoder_idx.items()}
categorical_values['County'] = list(county_subcounty_map.keys())

# Create ordered features list with type information
features_info = [
    {'name': feat, 'type': 'categorical' if feat in form_categorical_features else 'numerical'}
    for feat in top_features
]

# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict")
def predict_form():
    return render_template(
        "predict.html",
        features_info=features_info,
        categorical_values=categorical_values,
        county_subcounty_map=county_subcounty_map
    )

@app.route("/get_subcounties/<county>")
def get_subcounties(county):
    subcounties = county_subcounty_map.get(county, [])
    return jsonify(subcounties)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get('message', '').lower()
    
    # Simple AI responses based on keywords
    if 'predict' in user_message or 'prediction' in user_message:
        response = "To make a prediction, fill out all the form fields including your course, grades, and location details. The system will analyze your data and provide a care transition likelihood."
    elif 'score' in user_message or 'grade' in user_message:
        response = "Scores below 70 in any assessment may result in lower transition probability. A 'Fail' grade automatically results in 'Likely not to transition'."
    elif 'duration' in user_message or 'months' in user_message:
        response = "Minimum training duration: Childcare requires 2+ months, Eldercare requires 3+ months. Not meeting these requirements results in 'Likely not to transition'."
    elif 'county' in user_message or 'location' in user_message:
        response = "Select your county first, then the subcounty dropdown will automatically populate with relevant options for your area."
    elif 'course' in user_message:
        response = "Choose between Childcare or Eldercare courses. Each has different minimum duration requirements for successful transition."
    elif 'help' in user_message or 'how' in user_message:
        response = "I can help you understand the prediction form, scoring requirements, duration rules, and location selection. What specific question do you have?"
    else:
        response = "I'm here to help with the Care Transition Prediction system. Ask me about form fields, scoring requirements, duration rules, or how predictions work!"
    
    return jsonify({'response': response})

@app.route("/result", methods=["POST"])
def result():
    input_data = {}
    for feat in top_features:
        val = request.form.get(feat)
        if feat in form_categorical_features:
            input_data[feat] = val if val else ""
        else:
            input_data[feat] = float(val) if val else 0.0
    
    # Business Rules - Check BEFORE encoding (use original string values)
    score_features = ['Final Score', 'Practical-Icare', 'Average Attendance', 
                     'Average Practicals', 'Theory Exam - Icare', 'Hospital Internship Score', 
                     'Average Cats']
    
    # Check if any score is below 70
    low_scores = []
    for feature in score_features:
        if feature in input_data:
            score = input_data[feature]
            if isinstance(score, (int, float)) and score < 70:
                low_scores.append(feature)
    
    # Check if Final Grade is 'Fail'
    final_grade_fail = input_data.get('Final Grade') == 'Fail'
    
    # Check course duration requirements
    course = input_data.get('Course', '')
    duration = input_data.get('Course Duration Months', 0)
    duration_fail = False
    duration_reason = ""
    
    if course == 'Childcare' and float(duration) < 2:
        duration_fail = True
        duration_reason = "For Childcare, the minimum training duration should be 2 months which was not met"
    elif course == 'Eldercare' and float(duration) < 3:
        duration_fail = True
        duration_reason = "The minimum training time for Eldercare course is 3 months, which was not met with the caregiver"
    
    # Initialize reasons list
    failure_reasons = []
    
    # Apply business rules in order of severity - DURATION IS HIGHEST PRIORITY
    
    if duration_fail:
        # Duration requirement not met - OVERRIDES EVERYTHING (even good grades)
        will_transition = False
        probability = 0.05
        failure_reasons.append(duration_reason)
    elif final_grade_fail:
        # Final grade failure - cannot transition
        will_transition = False
        probability = 0.10
        failure_reasons.append("Final Grade is 'Fail'")
    elif low_scores:
        # Low scores but no fail - can transition with low probability
        will_transition = True
        probability = 0.25  # Low but possible transition
        failure_reasons.append(f"Low performance in: {', '.join(low_scores)} (below 70)")
    else:
        # All business rules pass - use ML model
        input_df = pd.DataFrame([input_data], columns=top_features)
        
        # Encode categorical features
        for col in form_categorical_features:
            if col in input_df.columns and col in cat_to_encoder_idx:
                idx = cat_to_encoder_idx[col]
                input_df[col] = encoders[idx].transform(input_df[col].astype(str))
        
        # Reorder columns to match scaler's expected order
        scaler_feature_order = scaler.feature_names_in_.tolist()
        input_df_scaled = input_df[scaler_feature_order]
        
        # Scale all features
        input_df_scaled = pd.DataFrame(scaler.transform(input_df_scaled), columns=scaler_feature_order)
        
        # Reorder back to top_features order for model
        input_df = input_df_scaled[top_features]
        
        # Use ML model prediction
        pred_prob = model.predict_proba(input_df)[:, 1][0]
        pred = model.predict(input_df)[0]
        probability = round(pred_prob, 4)
        will_transition = pred == 1
    
    return render_template(
        "result.html",
        will_transition=will_transition,
        probability=probability,
        failure_reasons=failure_reasons if 'failure_reasons' in locals() else [],
        low_scores=low_scores if 'low_scores' in locals() else [],
        final_grade_fail=final_grade_fail if 'final_grade_fail' in locals() else False,
        duration_fail=duration_fail if 'duration_fail' in locals() else False
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
