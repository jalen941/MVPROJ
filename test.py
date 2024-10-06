import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your model
model = load_model('cut_bruise.h5')

# Load and preprocess a new image
img_path = 'cut34.png'  # Change this to the path of your image
img = image.load_img(img_path, target_size=(128, 128))  # Resize to match input shape
img_array = image.img_to_array(img)  # Convert image to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize pixel values

# Make predictions
predictions = model.predict(img_array)

# Print raw predictions
print("Raw Predictions:", predictions)

# Convert predictions to class index
predicted_class_index = np.argmax(predictions, axis=1)

# Define your class names correctly
class_names = ['bruise', 'cut']  # This should match the training class order
#print(f'Predicted Class Index: {predicted_class_index[0]}')

# Set confidence thresholds
low_confidence_threshold = 0.5  # Below this value, classify as "neither"
high_confidence_threshold = 0.999  # Above this value, handle specifically

# Check the confidence level
max_prediction = np.max(predictions)
if max_prediction < low_confidence_threshold:
    print("Predicted Class: neither ")
elif max_prediction > high_confidence_threshold:
    print("Predicted Class: neither ")
else:
    print(f'Predicted Class: {class_names[predicted_class_index[0]]}')
