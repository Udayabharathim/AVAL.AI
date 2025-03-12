import os
import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model (try 'yolov8m.pt' or 'yolov8l.pt' for better detection)
model = YOLO("yolov8m.pt")  

# Define dataset path for pigmentation
pigmentation_path = "C:\\Users\\Elitebook 840 G6\\Documents\\AVAL2\\data\\pigmentation"

def auto_label_pigmentation(dataset_path):
    """ Auto-label pigmentation images using YOLOv8 and save bounding boxes as YOLO labels """
    print("\n🔍 Processing Pigmentation Dataset...")

    for split in ["train", "test", "valid"]:  # Ensure all folders are processed
        img_dir = os.path.join(dataset_path, split, "images")
        label_dir = os.path.join(dataset_path, split, "labels")
        os.makedirs(label_dir, exist_ok=True)

        if not os.path.exists(img_dir):
            print(f"🚫 Skipping {split}: No images folder found for pigmentation!")
            continue

        for img_name in os.listdir(img_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(img_dir, img_name)
                img = cv2.imread(img_path)

                # Run YOLO detection
                results = model(img)
                label_txt = os.path.join(label_dir, img_name.replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt"))

                with open(label_txt, "w") as f:
                    detected = False  
                    
                    for result in results:
                        for box in result.boxes:
                            x, y, w, h = box.xywh[0]  
                            class_id = 0  # Assign '0' for pigmentation
                            conf = box.conf[0]  

                            # Normalize coordinates
                            img_height, img_width, _ = img.shape
                            x /= img_width
                            y /= img_height
                            w /= img_width
                            h /= img_height

                            # Save in YOLO format
                            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
                            detected = True

                    # If no objects were detected, create an empty label file
                    if not detected:
                        f.write("")
                        print(f"⚠️ No detection for {img_name}, empty label file created.")

                print(f"✅ Labeled {img_name} -> {label_txt}")

# Run auto-labeling for pigmentation dataset
auto_label_pigmentation(pigmentation_path)

print("🎯 Pigmentation Auto-labeling Completed!")
