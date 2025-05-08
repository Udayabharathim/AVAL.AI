from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
import numpy as np
import os
import cv2
import joblib
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load models
print("Initializing models...")
acne_model = YOLO(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\runs\detect\train\weights\best.pt")
pcos_model = joblib.load('pcos_model.pkl')
model_features = joblib.load('model_features.pkl')

# Blood group mapping
blood_group_mapping = {
    'A+': 11, 'A-': 12, 'B+': 13, 'B-': 14,
    'O+': 15, 'O-': 16, 'AB+': 17, 'AB-': 18
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_acne(image_path):
    results = acne_model(image_path, conf=0.2)
    for result in results:
        for box in result.boxes:
            if result.names[int(box.cls[0].item())] == "acnes":
                return True
    return False

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        try:
            # Validate files
            if 'acne_image' not in request.files:
                return "Please upload the images", 400

            acne_file = request.files['acne_image']

            if not acne_file.filename:
                return "Please select both images", 400

            if not allowed_file(acne_file.filename):
                return "Only JPG, JPEG, PNG files allowed", 400

            # Process images
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            acne_path = os.path.join(app.config['UPLOAD_FOLDER'], acne_file.filename)
            acne_file.save(acne_path)
            acne_result = detect_acne(acne_path)

            # Process form data
            form_data = request.form
            required_fields = [
                'age', 'weight_now', 'weight_3_months_ago', 'height',
                'blood_group', 'months_between_periods', 'period_duration',
                'hair_loss', 'fast_food', 'exercise', 'mood_swings',
                'regular_periods', 'skin_darkening', 'excess_facial_hair'
            ]

            if any(field not in form_data for field in required_fields):
                return "Missing form fields", 400

            # Feature calculations
            age = float(form_data['age'])
            weight_now = float(form_data['weight_now'])
            weight_diff = weight_now - float(form_data['weight_3_months_ago'])
            height = float(form_data['height'])
            bmi = weight_now / ((height / 100) ** 2)

            input_data = pd.DataFrame([{
                'Age (yrs)': age,
                'Weight (Kg)': weight_now,
                'Height(Cm)': height,
                'BMI': round(bmi, 2),
                'Blood Group': blood_group_mapping.get(form_data['blood_group'], 0),
                'Cycle(R/I)': 1 if form_data['regular_periods'].lower() == 'yes' else 0,
                'Cycle length(days)': int(form_data['period_duration']),
                'Weight gain(Y/N)': 1 if weight_diff > 3 else 0,
                'hair growth(Y/N)': 1 if form_data['excess_facial_hair'].lower() == 'yes' else 0,
                'Skin darkening (Y/N)': 1 if form_data['skin_darkening'].lower() == 'yes' else 0,
                'Hair loss(Y/N)': 1 if form_data['hair_loss'].lower() == 'yes' else 0,
                'Pimples(Y/N)': 1 if acne_result else 0,
                'Fast food (Y/N)': 1 if form_data['fast_food'].lower() == 'yes' else 0,
                'Reg.Exercise(Y/N)': 1 if form_data['exercise'].lower() == 'yes' else 0
            }], columns=model_features)

            # Make prediction
            form_prediction = pcos_model.predict(input_data)[0]
            form_result = "PCOS Detected" if form_prediction == 1 else "No PCOS Detected"

            return redirect(url_for('result',
                form_result=form_result,
                acne_detected="Yes" if acne_result else "No",
                hirsutism_detected=form_data['excess_facial_hair'].capitalize(),
                skin_darkening_detected=form_data['skin_darkening'].capitalize()
            ))

        except Exception as e:
            print(f"Processing error: {str(e)}")
            return f"Error processing request: {str(e)}", 500

    return render_template('chatbot.html')

@app.route('/result')
def result():
    return render_template('result.html',
        form_result=request.args.get('form_result', 'N/A'),
        acne_detected=request.args.get('acne_detected', 'N/A'),
        hirsutism_detected=request.args.get('hirsutism_detected', 'N/A'),
        skin_darkening_detected=request.args.get('skin_darkening_detected', 'N/A')
    )

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)
