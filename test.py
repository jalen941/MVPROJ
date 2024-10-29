import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('cut_bruise_vgg16_transfer.h5')

img_path = "testImg/cutbig1_severe.jpg"
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0


predictions = model.predict(img_array)


print("Raw Predictions:", predictions)


predicted_class_index = np.argmax(predictions, axis=1)


class_names = ['bruise', 'cut']
#print(f'Predicted Class Index: {predicted_class_index[0]}')


low_confidence_threshold = 0.7
high_confidence_threshold = 0.999


max_prediction = np.max(predictions)
if max_prediction < low_confidence_threshold:
    print("Predicted Class: neither ")
elif max_prediction > high_confidence_threshold:
    print("Predicted Class: neither ")
else:
    print(f'Predicted Class: {class_names[predicted_class_index[0]]}')



import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_severity(width, height):
    area = width * height
    if area > 100000:
        return "Severe"
    elif area > 50000:
        return "Medium"
    else:
        return "Minor"

model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\Jalen\PycharmProjects\MVPROJ1\MVPROJ\yolov5\runs\train\injury_detection8\weights\best.pt')


img = Image.open(img_path)
img_width, img_height = img.size

results = model(img, size=320)


results.print()
results.show()


boxes = results.pred[0][results.pred[0][:, 4] >= 0.25]  # Filter with confidence >= 0.25


img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


for *xyxy, conf, cls in boxes.tolist():
    x1, y1, x2, y2 = map(int, xyxy)
    width = x2 - x1
    height = y2 - y1
    severity = get_severity(width, height)
    print("severity is", severity)
    print(f'Class: {model.names[int(cls)]}, Confidence: {conf:.2f}, Width: {width}, Height: {height}')

    cv2.rectangle(img_cv, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
    cv2.putText(img_cv, f'{model.names[int(cls)]}: {conf:.2f}', (int(xyxy[0]), int(xyxy[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
