import pandas as pd
import joblib
import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys


# Mapping for blood group values
blood_group_mapping = {
    'A+': 11, 'A-': 12, 'B+': 13, 'B-': 14,
    'O+': 15, 'O-': 16, 'AB+': 17, 'AB-': 18
}

# Mapping user input column names to match trained model feature names
column_mapping = {
    "Age (in Years)": "age_in_years",
    "Weight (in Kg)": "weight_in_kg",
    "Height (in Cm / Feet)": "height",
    "Can you tell us your blood group ?": "blood_group",
    "After how many months do you get your periods?\n(select 1- if every month/regular)": "period_cycle",
    "Have you gained weight recently?": "weight_gain",
    "Do you have excessive body/facial hair growth ?": "hair_growth",
    "Are you noticing skin darkening recently?": "skin_darkening",
    "Do have hair loss/hair thinning/baldness ?": "hair_loss",
    "Do you have pimples/acne on your face/jawline ?": "acne",
    "Do you eat fast food regularly ?": "fast_food",
    "Do you exercise on a regular basis ?": "exercise",
    "Do you experience mood swings ?": "mood_swings",
    "Are your periods regular ?": "are_your_periods_regular_",
    "How long does your period last ?": "period_duration"
}

def get_user_inputs():
    """Collects user inputs required for PCOS prediction."""
    inputs = {}
    try:
        inputs["Age (in Years)"] = int(input("Enter your age (in years): "))
        inputs["Weight (in Kg)"] = float(input("Enter your weight (in kg): "))
        inputs["Height (in Cm / Feet)"] = float(input("Enter your height (in cm): "))
        
        blood_group = input("Enter your blood group (A+, A-, B+, B-, O+, O-, AB+, AB-): ").strip().upper()
        inputs["Can you tell us your blood group ?"] = blood_group_mapping.get(blood_group, -1)
        
        inputs["After how many months do you get your periods?\n(select 1- if every month/regular)"] = int(input("Enter period cycle (1 if regular, otherwise specify months): "))
        
        def yes_no_input(prompt):
            return 1 if input(prompt + " (yes/no): ").strip().lower() == "yes" else 0
        
        inputs["Have you gained weight recently?"] = yes_no_input("Have you gained weight recently?")
        inputs["Do you have excessive body/facial hair growth ?"] = yes_no_input("Do you have excessive body/facial hair growth?")
        inputs["Are you noticing skin darkening recently?"] = yes_no_input("Are you noticing skin darkening?")
        inputs["Do have hair loss/hair thinning/baldness ?"] = yes_no_input("Do you have hair loss/baldness?")
        inputs["Do you have pimples/acne on your face/jawline ?"] = yes_no_input("Do you have pimples/acne?")
        inputs["Do you eat fast food regularly ?"] = yes_no_input("Do you eat fast food regularly?")
        inputs["Do you exercise on a regular basis ?"] = yes_no_input("Do you exercise regularly?")
        inputs["Do you experience mood swings ?"] = yes_no_input("Do you experience mood swings?")
        inputs["Are your periods regular ?"] = yes_no_input("Are your periods regular?")
        inputs["How long does your period last ?"] = int(input("How long does your period last (in days)? "))
    except ValueError:
        print("❌ Invalid input. Please enter the correct data types.")
        return None

    user_data = pd.DataFrame([inputs])
    user_data.rename(columns=column_mapping, inplace=True)
    return user_data

def capture_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture image")
            break
        frame = cv2.flip(frame, 1)
        cv2.imshow("Press SPACE to Capture | ESC to Exit", frame)
        key = cv2.waitKey(1)
        if key == 32:
            image_path = "user_image.jpg"
            cv2.imwrite(image_path, frame)
            print(f"📸 Image captured and saved as {image_path}")
            cap.release()
            cv2.destroyAllWindows()
            return image_path
        elif key == 27:
            print("❌ Capture cancelled")
            break
    cap.release()
    cv2.destroyAllWindows()
    return None

def upload_image():
    app = QApplication(sys.argv)
    file_path, _ = QFileDialog.getOpenFileName(None, "Select an Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
    
    if file_path:
        print(f"📁 Image uploaded: {file_path}")
        return file_path
    else:
        print("❌ No file selected.")
        return None
    
def predict_pcos(user_data):
    try:
        model = joblib.load("pcos_model.pkl")
        user_data = user_data.reindex(columns=model.feature_names_in_, fill_value=0)
        prediction = model.predict(user_data)[0]
        return prediction
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def detect_acne_pigmentation(image_path):
    try:
        acne_model = YOLO(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\runs\detect\train\weights\best.pt")
        hirsutism_model = YOLO(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\runs\detect\train2\weights\best.pt")
        
        acne_results = acne_model(image_path, conf=0.2)
        hirsutism_results = hirsutism_model(image_path, conf=0.2)

        acne_detected = any(len(result.boxes) > 0 for result in acne_results)
        hirsutism_detected = any(len(result.boxes) > 0 for result in hirsutism_results)

        return acne_detected, hirsutism_detected
    except Exception as e:
        print(f"❌ Error in YOLO model detection: {e}")
        return False, False

def main():
    user_data = get_user_inputs()
    if user_data is None:
        return
    
    pcos_prediction = predict_pcos(user_data)
    if pcos_prediction is None:
        return
    
    image_path = None
    if input("Do you want to capture or upload an image for acne detection? (capture/upload/no): ").strip().lower() == "capture":
        image_path = capture_image()
    elif input("Do you want to upload an image instead? (yes/no): ").strip().lower() == "yes":
        image_path = upload_image()
    
    if image_path:
        acne_detected, hirsutism_detected = detect_acne_pigmentation(image_path)
        if acne_detected or hirsutism_detected:
            pcos_prediction = 1  
    
    print("You may have PCOS. Please consult a doctor." if pcos_prediction == 1 else "PCOS is unlikely based on the given data.")

if __name__ == "__main__":
    main()
