import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('cut_bruise_vgg16_transfer.h5')
test_img_dir = r"C:\Users\Jalen\PycharmProjects\MVPROJ1\MVPROJ\testImg"

class_names = ['bruise', 'cut']

low_confidence_threshold = 0.61
high_confidence_threshold = 0.999

correct_predictions = 0
total_images = 0

for img_name in os.listdir(test_img_dir):

    img_path = os.path.join(test_img_dir, img_name)


    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    max_prediction = np.max(predictions)
    if max_prediction < low_confidence_threshold:
        predicted_class = "neither"
    elif max_prediction > high_confidence_threshold:
        predicted_class = "neither"
    else:
        predicted_class = class_names[predicted_class_index]

    if "cut" in img_name.lower():
        true_class = "cut"
    elif "bruise" in img_name.lower():
        true_class = "bruise"
    elif "none" in img_name.lower():
        true_class = "neither"
    else:
        true_class = None
        continue


    if predicted_class == true_class:
        correct_predictions += 1
    else:
        print("predicted", true_class, "wrong")

    total_images += 1


if total_images > 0:
    accuracy = correct_predictions / total_images * 100
    print(f'Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_images} correct predictions)')
else:
    print("No images found in the directory or no valid labels detected.")
