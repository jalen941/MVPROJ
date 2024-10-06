import os
import shutil
from pycocotools.coco import COCO

# Load COCO annotations file
coco = COCO('injury.v2i.coco-segmentation/train/_annotations.coco.json')

# Path to images folder
img_dir = 'injury.v2i.coco-segmentation/train'

# Create separate folders for each class
output_dir = 'dataset/'
class_names = ['Cut', 'bruise']  # Specify your target classes here
class_ids = []  # List to hold category IDs of interest

# Create directories for each class if they don't exist
for class_name in class_names:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

# Get category IDs of interest
for class_name in class_names:
    category_id = coco.getCatIds(catNms=[class_name])
    if category_id:
        class_ids.append(category_id[0])  # Assuming each class has a unique ID

# Now let's iterate over each image and ignore those not belonging to desired classes
for img_id in coco.getImgIds():
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=class_ids, iscrowd=False)

    if not ann_ids:  # Skip images without annotations in target categories
        continue

    # Load image information
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, img_info['file_name'])

    # Get the first annotation category (if there are multiple categories, we simplify here)
    ann_info = coco.loadAnns(ann_ids)[0]
    class_id = ann_info['category_id']

    # Find class name based on category_id
    class_name = coco.loadCats([class_id])[0]['name']

    if class_name in class_names:
        # Copy the image to the corresponding class folder
        shutil.copy(img_path, os.path.join(output_dir, class_name, img_info['file_name']))
