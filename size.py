import cv2
import numpy as np

def detect_cut(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges for cuts (reddish)
    cut_lower1 = np.array([0, 70, 50])  # Lower bound for red
    cut_upper1 = np.array([10, 255, 255])  # Upper bound for red
    cut_lower2 = np.array([170, 70, 50])  # Include the higher red range
    cut_upper2 = np.array([180, 255, 255])

    # Create masks for cuts
    cut_mask1 = cv2.inRange(hsv, cut_lower1, cut_upper1)
    cut_mask2 = cv2.inRange(hsv, cut_lower2, cut_upper2)

    # Combine masks
    cut_mask = cv2.bitwise_or(cut_mask1, cut_mask2)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    cut_mask = cv2.morphologyEx(cut_mask, cv2.MORPH_CLOSE, kernel)
    cut_mask = cv2.morphologyEx(cut_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(cut_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Filter contours to find the largest one
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding box around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Dynamic padding based on contour size
        padding_factor = min(0.1, 0.05 * (1 + np.log(w * h)))  # Adjust based on area
        x += int(w * padding_factor)
        y += int(h * padding_factor)
        w -= int(2 * w * padding_factor)
        h -= int(2 * h * padding_factor)

        # Draw the bounding box on the original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red box

        # Show the mask and original image with bounding box for debugging
        cv2.imshow("Cut Mask", cut_mask)
        cv2.imshow("Detected Cut", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return (x, y, w, h)  # Return coordinates and size of the detected cut
    else:
        print("No contours found")
        return None

# Example usage
img_path = 'testImg/bruise4_minor.jpg'  # Update with the path to your image
detect_cut(img_path)
