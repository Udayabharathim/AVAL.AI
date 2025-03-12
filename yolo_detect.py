import cv2
from ultralytics import YOLO

# Load the trained YOLO models
acne_model = YOLO(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\runs\detect\train\weights\best.pt")  # Acne model
hirsutism_model = YOLO(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\runs\detect\train2\weights\best.pt")  # Hirsutism model

def detect_conditions(image_path):
    """
    Detects acne and hirsutism in the given image.
    Returns:
        acne_detected (int): 1 if acne is detected, else 0.
        hirsutism_detected (int): 1 if hirsutism is detected, else 0.
    """
    # Run YOLO detection for acne
    acne_results = acne_model(image_path, conf=0.2)
    acne_detected = 0
    for result in acne_results:
        for box in result.boxes:
            if result.names[int(box.cls[0].item())] == "acne":
                acne_detected = 1
                break

    # Run YOLO detection for hirsutism
    hirsutism_results = hirsutism_model(image_path, conf=0.2)
    hirsutism_detected = 0
    for result in hirsutism_results:
        for box in result.boxes:
            if result.names[int(box.cls[0].item())] == "hirsutism":
                hirsutism_detected = 1
                break

    return acne_detected, hirsutism_detected