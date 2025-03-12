from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load your trained YOLO model (Change the path to your actual model)
model = YOLO("C:\\Users\\Elitebook 840 G6\\Documents\\AVAL2\\yolov8n.pt")  # Replace with your trained model path

# Load and display the test image
image_path = "C:\\Users\\Elitebook 840 G6\\Documents\\AVAL2\\data\\acne\\test\\images\\acne-15_jpeg.rf.510f6ab6153777e8087f1439ef39d8cc.jpg"  # Replace with the test image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper visualization

# Run YOLO detection
results = model(image)

# Display the image with detected bounding boxes
for result in results:
    result.show()  # Show detection results

# Optional: Save the output image
results[0].save("C:\\Users\\Elitebook 840 G6\\Documents\\AVAL2\\result.jpg")  # Change path if needed