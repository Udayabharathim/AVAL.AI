from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
from ultralytics import YOLO

app = Flask(__name__)

# Load the saved model
print("Loading the saved model...")
model = joblib.load("pcos_model.pkl")
print("Model loaded successfully! Model details:", model)

# Load YOLO models for acne and hirsutism detection
print("Loading YOLO models...")
acne_model = YOLO(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\runs\detect\train\weights\best.pt")  # Acne detection
hirsutism_model = YOLO(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\runs\detect\train2\weights\best.pt")  # Hirsutism detection
skin_darkening_model = YOLO(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\runs\detect\train3\weights\best.pt")  # Skin darkening detection
print("YOLO models loaded successfully!")

# Blood group mapping
blood_group_mapping = {
    'A+': 11, 'A-': 12,
    'B+': 13, 'B-': 14,
    'O+': 15, 'O-': 16,
    'AB+': 17, 'AB-': 18
}

def detect_features(image_path):
    """
    Detects acne, hirsutism (excess facial/body hair), and skin darkening in the given image.
    Returns:
        acne_detected (int): 1 if acne is detected, else 0.
        hirsutism_detected (int): 1 if excess body/facial hair is detected, else 0.
        skin_darkening_detected (int): 1 if skin darkening is detected, else 0.
    """
    acne_detected = 0
    hirsutism_detected = 0
    skin_darkening_detected = 0

    # YOLO detection for acne
    print("Running YOLO detection for acne...")
    acne_results = acne_model(image_path, conf=0.2)
    for result in acne_results:
        for box in result.boxes:
            if result.names[int(box.cls[0].item())] == "acne":
                acne_detected = 1
                break

    # YOLO detection for hirsutism
    print("Running YOLO detection for excess facial/body hair...")
    hirsutism_results = hirsutism_model(image_path, conf=0.2)
    for result in hirsutism_results:
        for box in result.boxes:
            if result.names[int(box.cls[0].item())] == "hirsutism":
                hirsutism_detected = 1
                break

    # YOLO detection for skin darkening
    print("Running YOLO detection for skin darkening...")
    skin_darkening_results = skin_darkening_model(image_path, conf=0.2)
    for result in skin_darkening_results:
        for box in result.boxes:
            if result.names[int(box.cls[0].item())] == "skin_darkening":
                skin_darkening_detected = 1
                break

    return acne_detected, hirsutism_detected, skin_darkening_detected

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        # Debug: Print form data
        print("Form Data Received:", request.form)

        # Get user input data
        age = float(request.form["What is your age (in years)?"])
        weight_now = float(request.form["What is your current weight (in kg)?"])
        weight_3_months_ago = float(request.form["What was your weight 3 months ago (in kg)?"])
        height = float(request.form["What is your height (in cm)?"])
        blood_group = request.form["What is your blood group?"]
        months_between_periods = int(request.form["How many months between your periods?"])

        # Compute weight gain status
        weight_diff = weight_now - weight_3_months_ago
        weight_gain_recently = 1 if weight_diff > 2 else 0  # Threshold can be adjusted

        # Handle yes/no questions
        hair_loss = 1 if request.form["Do you experience hair loss? (yes/no)"].lower() == "yes" else 0
        fast_food = 1 if request.form["Do you consume fast food regularly? (yes/no)"].lower() == "yes" else 0
        exercise = 1 if request.form["Do you exercise regularly? (yes/no)"].lower() == "yes" else 0
        mood_swings = 1 if request.form["Do you experience mood swings? (yes/no)"].lower() == "yes" else 0
        are_your_periods_regular_ = 1 if request.form["Are your periods regular? (yes/no)"].lower() == "yes" else 0

        period_duration = int(request.form["What is your period duration (in days)?"])

        # Get the uploaded image for acne, hirsutism, and skin darkening detection
        image = request.files["image"]  # Use the correct key
        image_path = os.path.join("uploads", image.filename)
        image.save(image_path)
        print("Image uploaded successfully! Image path:", image_path)

        # Perform feature detection using YOLO
        acne_detected, hirsutism_detected, skin_darkening_detected = detect_features(image_path)
        print(f"Acne detected: {acne_detected}, Hirsutism detected: {hirsutism_detected}, Skin Darkening detected: {skin_darkening_detected}")

        # Map blood group to numeric value
        blood_group_numeric = blood_group_mapping.get(blood_group, 0)
        print("Blood Group Numeric Value:", blood_group_numeric)

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            "age_in_years": [age],
            "weight_in_kg": [weight_now],
            "height": [height],
            "bmi": [weight_now / (height / 100) ** 2],
            "blood_group": [blood_group_numeric],
            "months_between_periods": [months_between_periods],
            "weight_gain_recently": [weight_gain_recently],
            "excess_body_facial_hair": [hirsutism_detected],
            "skin_darkening": [skin_darkening_detected],
            "hair_loss": [hair_loss],
            "acne": [acne_detected],
            "fast_food": [fast_food],
            "exercise": [exercise],
            "mood_swings": [mood_swings],
            "are_your_periods_regular_": [are_your_periods_regular_],
            # Remove "period_duration" if it's not part of the training data
        })
        print("Input Data for Prediction:", input_data)

        # Ensure the input data has only the features the model was trained on
        # Remove "period_duration" if it's not part of the training data
        input_data = input_data.drop(columns=["period_duration"], errors="ignore")

        # Make PCOS prediction
        prediction = model.predict(input_data)[0]
        result = "PCOS Detected" if prediction == 1 else "No PCOS Detected"
        print(f"Prediction: {result}")

        return {
            "result": result,
            "acne_detected": "Yes" if acne_detected else "No",
            "hirsutism_detected": "Yes" if hirsutism_detected else "No",
            "skin_darkening_detected": "Yes" if skin_darkening_detected else "No"
        }

    return render_template("chatbot.html")

@app.route("/result")
def result():
    # Fetch query parameters from the URL
    result = request.args.get("result", "N/A")  # Default to "N/A" if missing
    acne_detected = request.args.get("acne_detected", "N/A")
    hirsutism_detected = request.args.get("hirsutism_detected", "N/A")
    skin_darkening_detected = request.args.get("skin_darkening_detected", "N/A")

    # Debug: Print the fetched data
    print(f"Result: {result}, Acne: {acne_detected}, Hirsutism: {hirsutism_detected}, Skin Darkening: {skin_darkening_detected}")

    # Pass the data to the result.html template
    return render_template(
        "result.html",
        result=result,
        acne_detected=acne_detected,
        hirsutism_detected=hirsutism_detected,
        skin_darkening_detected=skin_darkening_detected
    )

if __name__ == "__main__":
    # Create uploads directory if it doesn't exist
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)