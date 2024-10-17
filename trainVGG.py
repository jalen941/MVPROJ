import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up data generators for training and validation sets
train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'dataset/',  # Path to the dataset with subfolders for each class
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Load the VGG16 model with pre-trained ImageNet weights, exclude the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the layers of VGG16 so they are not trainable
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top for classification
model = Sequential([
    base_model,  # Use the pre-trained VGG16 model as the base
    Flatten(),  # Flatten the output of the convolutional layers
    Dense(128, activation='relu'),  # Add a fully connected layer
    Dropout(0.5),  # Add dropout for regularization
    Dense(2, activation='softmax')  # Output layer with 2 classes ('cut' and 'bruise')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10  # Adjust the number of epochs as needed
)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Save the model
model.save('cut_bruise_vgg16_transfer.h5')
