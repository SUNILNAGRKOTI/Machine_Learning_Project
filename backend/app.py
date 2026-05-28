from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
from datetime import datetime
import os
import traceback

app = Flask(__name__)
CORS(app)

rf_model = None
metadata = None
predictions_history = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_models():
    """ Load trained models and metadata """
    global rf_model, metadata
    try:
        rf_path = os.path.join(BASE_DIR, 'random_forest_model.pkl')
        meta_path = os.path.join(BASE_DIR, 'model_metadata.json')

        if os.path.exists(rf_path):
            rf_model = joblib.load(rf_path)
            print("✓ Random Forest Model loaded.")
        else:
            print("⚠️ Random Forest Model not found. Run main.py first.")

        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            print("✓ Model Metadata loaded.")

    except Exception as e:
        print("❌ Error loading models:", e)
        traceback.print_exc()

def classify_productivity(score):
    """
    Classify score into HIGH, MEDIUM, or LOW (0-100 scale).
    Thresholds adjusted based on achieved model performance.
    """
    # *** ADJUSTED THRESHOLDS ***
    # Score 75 or higher is classified as HIGH
    if score >= 75:
        return {'level': 'HIGH', 'emoji': '✅', 'color': 'high', 'description': 'Excellent productivity!'}
    # Score 45 or higher is classified as MEDIUM
    elif score >= 45:
        return {'level': 'MEDIUM', 'emoji': '⚠️', 'color': 'medium', 'description': 'Good productivity'}
    # Otherwise, it's LOW
    else:
        return {'level': 'LOW', 'emoji': '❌', 'color': 'low', 'description': 'Low productivity'}

def generate_recommendations(sleep, study, screen, activity_hours, caffeine):
    """ Generate personalized recommendations based on input """
    recs = []
    
    # Activity is passed as minutes from front-end, converted to hours for model
    activity_minutes = activity_hours * 60

    if sleep < 7.0:
        recs.append({'category': 'Sleep', 'message': f'😴 Increase sleep duration. Target 7-9 hours.', 'priority': 'high'})
    elif sleep > 9.0:
        recs.append({'category': 'Sleep', 'message': '💤 Avoid oversleeping. Max 9 hours is optimal.', 'priority': 'medium'})
    
    if study < 4.0:
        recs.append({'category': 'Study', 'message': '📚 Focus on increasing focused study time (4+ hours).', 'priority': 'high'})

    if screen > 3.5:
        recs.append({'category': 'Screen Time', 'message': f'📱 Reduce screen time. Current: {screen:.1f}h. It strongly impacts your focus.', 'priority': 'high'})

    if activity_minutes < 30:
        recs.append({'category': 'Physical Activity', 'message': f'🏃 Increase physical activity! Aim for 30-60 minutes daily.', 'priority': 'high'})
    elif activity_minutes > 120:
        recs.append({'category': 'Physical Activity', 'message': '🧘 Great activity level, but ensure rest is integrated.', 'priority': 'low'})

    if caffeine > 3.0:
        recs.append({'category': 'Caffeine', 'message': f'☕ High caffeine intake! Limit to 3 cups or less for better sleep quality.', 'priority': 'medium'})

    if not recs:
        recs.append({'category': 'Overall', 'message': '🎉 Your metrics are well-balanced! Keep up the great habits.', 'priority': 'low'})

    return recs

def calculate_feature_importance():
    """ Calculate normalized feature importance from metadata """
    if metadata and 'feature_importance' in metadata:
        imp_dict = metadata['feature_importance']
        total = sum(imp_dict.values())
        if total == 0: return {}
        
        # Normalize and convert to percentage
        return {k: round((v / total) * 100, 1) for k, v in imp_dict.items()}
    
    # Return a reasonable default if metadata is missing (e.g., for testing)
    return {'Sleep_Duration': 35.0, 'Study_Hours': 25.0, 'Screen_Time': 20.0, 'Physical_Activity': 15.0, 'Caffeine_Intake': 5.0}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required = ['sleep', 'study', 'screen', 'activity', 'caffeine']
        
        for field in required:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400

        sleep = float(data['sleep'])
        study = float(data['study'])
        screen = float(data['screen'])
        activity_minutes = float(data['activity'])
        caffeine = float(data['caffeine'])
        
        # --- CRITICAL: Convert activity from MINUTES (front-end) to HOURS (model) ---
        activity_hours = activity_minutes / 60

        if rf_model is None:
            return jsonify({'success': False, 'error': 'Model not loaded. Please run main.py first.'}), 500

        # Input data array must match the training features order
        input_arr = np.array([[sleep, study, screen, activity_hours, caffeine]])

        prediction = rf_model.predict(input_arr)[0]
        
        # Clip the score to the valid 0-100 range and round
        score = max(0.0, min(100.0, float(prediction)))
        score = round(score, 1)

        classification = classify_productivity(score)
        recommendations = generate_recommendations(sleep, study, screen, activity_hours, caffeine)
        feature_importance_raw = calculate_feature_importance()
        
        # Format feature importance keys to match front-end chart labels
        feature_importance = {
            'sleep': feature_importance_raw.get('Sleep_Duration', 0.0),
            'study': feature_importance_raw.get('Study_Hours', 0.0),
            'screen': feature_importance_raw.get('Screen_Time', 0.0),
            'activity': feature_importance_raw.get('Physical_Activity', 0.0),
            'caffeine': feature_importance_raw.get('Caffeine_Intake', 0.0),
        }
        
        model_accuracy = metadata.get('rf_r2', 0.85) if metadata else 0.85

        # Store prediction record
        record = {'timestamp': datetime.now().isoformat(), 'input': {'sleep': sleep, 'study': study, 'screen': screen, 'activity': activity_minutes, 'caffeine': caffeine}, 'output': {'score': score, 'level': classification['level']}}
        predictions_history.append(record)

        resp = {
            'success': True,
            'prediction': {'score': score, 'level': classification['level'], 'emoji': classification['emoji'], 'color': classification['color'], 'description': classification['description']},
            'confidence': int(round(model_accuracy * 100)),
            'recommendations': recommendations,
            'feature_importance': feature_importance,
        }
        return jsonify(resp), 200

    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid input values (must be numbers).'}), 400
    except Exception as e:
        print("Prediction Error:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Internal server error during prediction: {str(e)}'}), 500

@app.route('/history', methods=['GET'])
def history():
    return jsonify({'total_predictions': len(predictions_history), 'history': predictions_history[-50:]})


if __name__ == '__main__':
    load_models()
    # This ensures the Flask server starts ONLY after models are loaded.
    print("\nPRODUCTIVITY PREDICTION SYSTEM - FLASK API\nAPI running at: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
