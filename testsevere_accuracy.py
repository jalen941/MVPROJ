import os
import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\Jalen\PycharmProjects\MVPROJ1\MVPROJ\yolov5\runs\train\injury_detection8\weights\best.pt')

# Directory containing test images
test_img_dir = r"C:\Users\Jalen\PycharmProjects\MVPROJ1\MVPROJ\testImg"

# Thresholds for severity levels
def get_severity(width, height):
    area = width * height
    if area > 100000:
        return "Severe"
    elif area > 70000:
        return "Medium"
    else:
        return "Minor"

correct_predictions = 0
total_images = 0

for img_name in os.listdir(test_img_dir):
    img_path = os.path.join(test_img_dir, img_name)

    # Open image
    img = Image.open(img_path)
    img_width, img_height = img.size

    # Run model prediction
    results = model(img, size=320)

    # Filter out boxes with low confidence
    boxes = results.pred[0][results.pred[0][:, 4] >= 0.25]  # Filter with confidence >= 0.25

    predicted_severity = None

    if boxes.shape[0] > 0:
        # Loop through each detected box to determine severity
        for *xyxy, conf, cls in boxes.tolist():
            x1, y1, x2, y2 = map(int, xyxy)
            width = x2 - x1
            height = y2 - y1
            predicted_severity = get_severity(width, height)
            break  # Only using the first detected injury's severity for simplicity

    # Determine the true severity based on the file name
    img_name_lower = img_name.lower()
    if "severe" in img_name_lower:
        true_severity = "Severe"
    elif "medium" in img_name_lower:
        true_severity = "Medium"
    elif "minor" in img_name_lower:
        true_severity = "Minor"
    else:
        true_severity = None  # No injury in this image

    # Check if the prediction matches the true severity
    if true_severity == predicted_severity or (true_severity is None and predicted_severity is None):
        correct_predictions += 1
    else:
        print(f"Incorrect prediction for {img_name}: Predicted {predicted_severity}, True {true_severity}")

    total_images += 1

# Calculate and print accuracy
if total_images > 0:
    accuracy = correct_predictions / total_images * 100
    print(f'Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_images} correct predictions)')
else:
    print("No images found in the directory or no valid labels detected.")
