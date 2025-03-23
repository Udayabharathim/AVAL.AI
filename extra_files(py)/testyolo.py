from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load your trained YOLO models (Change the paths to your actual models)
acne_model = YOLO(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\runs\detect\train\weights\best.pt")  # Replace with your acne model path
pigmentation_model = YOLO(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\runs\detect\train2\weights\best.pt")  # Replace with your pigmentation model path
hirsutism_model = YOLO(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\runs\detect\train\weights\best.pt")  # Replace with your hirsutism model path

# Load and display the test image
image_path = r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\pigmentation\test\images\melasma24_jpg.rf.f2fec9ba6bb588eb0fd3ee02eb33eb70.jpg"  # Replace with the test image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper visualization

# Run YOLO detection for each model
acne_results = acne_model(image)
pigmentation_results = pigmentation_model(image)
hirsutism_results = hirsutism_model(image)

# Function to display image with detections using matplotlib
def display_results(results, title):
    for result in results:
        # Plot the results using matplotlib
        plt.figure()
        plt.imshow(result.plot())  # Plot the image with detections
        plt.title(title)  # Set the title
        plt.axis('off')  # Hide axes
        plt.show()

# Display the results for each model
display_results(acne_results, "Acne Detection")
display_results(pigmentation_results, "Pigmentation Detection")
display_results(hirsutism_results, "Hirsutism Detection")

# Optional: Save the output images
acne_results[0].save("C:\\Users\\Elitebook 840 G6\\Documents\\AVAL2\\acne_result.jpg")  # Change path if needed
pigmentation_results[0].save("C:\\Users\\Elitebook 840 G6\\Documents\\AVAL2\\pigmentation_result.jpg")  # Change path if needed
hirsutism_results[0].save("C:\\Users\\Elitebook 840 G6\\Documents\\AVAL2\\hirsutism_result.jpg")  # Change path if needed