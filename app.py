from flask import Flask, request, render_template, redirect, url_for, session
import joblib
import pandas as pd
import os
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load the saved model
print("Loading the saved model...")
model = joblib.load("pcos_ensemble_model.pkl")
print("Model loaded successfully! Model details:", model)

# Load YOLO model for acne detection only
print("Loading YOLO model...")
acne_model = YOLO(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\runs\detect\train\weights\best.pt")
print("YOLO model loaded successfully!")

# Blood group mapping
blood_group_mapping = {
    'A+': 11, 'A-': 12,
    'B+': 13, 'B-': 14,
    'O+': 15, 'O-': 16,
    'AB+': 17, 'AB-': 18
}

def detect_acne(image_path):
    """Detects acne in the given image using YOLO model"""
    print("Running YOLO detection for acne...")
    acne_results = acne_model(image_path, conf=0.2)
    for result in acne_results:
        for box in result.boxes:
            if result.names[int(box.cls[0].item())] == "acne":
                return 1
    return 0

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        # Get user input data
        form_data = request.form
        
        # Basic information
        age = float(form_data["What is your age (in years)?"])
        weight_now = float(form_data["What is your current weight (in kg)?"])
        weight_3_months_ago = float(form_data["What was your weight 3 months ago (in kg)?"])
        height = float(form_data["What is your height (in cm)?"])
        blood_group = form_data["What is your blood group?"]
        months_between_periods = int(form_data["How many months between your periods?"])
        period_duration = int(form_data["What is your period duration (in days)?"])

        # Compute weight gain status
        weight_diff = weight_now - weight_3_months_ago
        weight_gain_recently = 1 if weight_diff > 2 else 0

        # Yes/No questions
        hair_loss = 1 if form_data["Do you experience hair loss? (yes/no)"].lower() == "yes" else 0
        fast_food = 1 if form_data["Do you consume fast food regularly? (yes/no)"].lower() == "yes" else 0
        exercise = 1 if form_data["Do you exercise regularly? (yes/no)"].lower() == "yes" else 0
        mood_swings = 1 if form_data["Do you experience extreme mood swings? (yes/no)"].lower() == "yes" else 0
        regular_periods = 1 if form_data["Are your periods regular? (yes/no)"].lower() == "yes" else 0
        
        # New yes/no questions
        skin_darkening = 1 if form_data["Do you experience skin darkening? (yes/no)"].lower() == "yes" else 0
        excess_facial_hair = 1 if form_data["Do you have excess facial/body hair? (yes/no)"].lower() == "yes" else 0

        # Acne detection from image
        image = request.files["image"]
        image_path = os.path.join("uploads", image.filename)
        image.save(image_path)
        acne_detected = detect_acne(image_path)

        # Map blood group to numeric value
        blood_group_numeric = blood_group_mapping.get(blood_group, 0)

        # Create input DataFrame
        input_data = pd.DataFrame({
            "age_in_years": [age],
            "weight_in_kg": [weight_now],
            "height": [height],
            "bmi": [weight_now / (height / 100) ** 2],
            "blood_group": [blood_group_numeric],
            "months_between_periods": [months_between_periods],
            "weight_gain_recently": [weight_gain_recently],
            "excess_body_facial_hair": [excess_facial_hair],
            "skin_darkening": [skin_darkening],
            "hair_loss": [hair_loss],
            "acne": [acne_detected],
            "fast_food": [fast_food],
            "exercise": [exercise],
            "mood_swings": [mood_swings],
            "are_your_periods_regular_": [regular_periods],
        })

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "PCOS Detected" if prediction == 1 else "No PCOS Detected"

        return {
            "result": result,
            "acne_detected": "Yes" if acne_detected else "No",
            "hirsutism_detected": "Yes" if excess_facial_hair else "No",
            "skin_darkening_detected": "Yes" if skin_darkening else "No"
        }

    return render_template("chatbot.html")


@app.route("/result")
def result():
    # Store PCOS detection result in session
    result_status = request.args.get("result", "N/A")
    session['pcos_detected'] = (result_status == "PCOS Detected")
    
    return render_template(
        "result.html",
        result=result_status,
        acne_detected=request.args.get("acne_detected", "N/A"),
        hirsutism_detected=request.args.get("hirsutism_detected", "N/A"),
        skin_darkening_detected=request.args.get("skin_darkening_detected", "N/A")
    )

@app.route("/recommendations", methods=["GET", "POST"])
def recommendations():
    if 'pcos_detected' not in session:
        return redirect(url_for('home'))

    if request.method == "POST":
        # Get user preferences
        breakfast_time = request.form.get('breakfast_time', 'less_time')
        exercise_preference = request.form.get('exercise_preference', 'morning')
        cooking_time = request.form.get('cooking_time', '15')

        # Generate recommendations
        diet_plan = generate_diet_plan(
            session['pcos_detected'],
            breakfast_time,
            int(cooking_time)
        )
        exercise_plan = generate_exercise_plan(
            session['pcos_detected'],
            exercise_preference
        )

        return render_template("recommendations.html",
                             show_plan=True,
                             diet_plan=diet_plan,
                             exercise_plan=exercise_plan)

    return render_template("recommendations.html", show_plan=False)

def generate_diet_plan(pcos_detected, breakfast_time, cooking_time):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    plan = []
    
    for day in days:
        if pcos_detected:
            breakfast = "Smoothie with spinach, berries, and almond milk" if breakfast_time == 'less_time' else "Vegetable omelette with avocado"
            lunch = "Grilled chicken salad with olive oil dressing"
            dinner = "Baked salmon with quinoa and steamed broccoli"
        else:
            breakfast = "Overnight oats with fruits" if breakfast_time == 'less_time' else "Whole grain pancakes with berries"
            lunch = "Quinoa bowl with mixed vegetables"
            dinner = "Grilled chicken with brown rice and asparagus"

        if cooking_time < 15:
            breakfast = "Greek yogurt with nuts and seeds"

        plan.append({
            "day": day,
            "breakfast": breakfast,
            "lunch": lunch,
            "dinner": dinner
        })
    return plan

def generate_exercise_plan(pcos_detected, preference):
    plan = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    for i, day in enumerate(days):
        exercises = []
        if preference in ['morning', 'both']:
            exercises.append("Morning: 30-min " + 
                ("yoga" if pcos_detected else "cardio"))
        if preference in ['evening', 'both']:
            exercises.append("Evening: " +
                ("strength training" if pcos_detected else "pilates"))
        
        if not exercises:
            exercises.append("Rest day")
            
        plan.append({
            "day": day,
            "exercises": " + ".join(exercises)
        })
    return plan

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)