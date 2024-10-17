

---

# Skin Cancer Detection Model Documentation

This project uses Convolutional Neural Networks (CNNs) built with Python, TensorFlow, and Keras to classify skin cancer images into multiple categories, such as melanoma, nevus, basal cell carcinoma, and more. The dataset includes several skin cancer types, organized into distinct folders for each category.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Project Setup](#project-setup)
   - [Install Python](#install-python)
   - [Install VS Code](#install-vs-code)
   - [Install Python Extension for VS Code](#install-python-extension-for-vs-code)
3. [Creating a Virtual Environment](#creating-a-virtual-environment)
4. [Installing Required Libraries](#installing-required-libraries)
5. [Dataset](#dataset)
6. [Data Preprocessing](#data-preprocessing)
7. [Building the CNN Model](#building-the-cnn-model)
8. [Training the Model](#training-the-model)
9. [Evaluating the Model](#evaluating-the-model)
10. [Saving and Loading the Model](#saving-and-loading-the-model)
11. [Making Predictions](#making-predictions)
12. [Running the Code](#running-the-code)
13. [Optional: Deployment](#optional-deployment)

---

## Prerequisites

Before starting, ensure you have the following installed:
- Python 3.7+
- Visual Studio Code (VS Code)
- TensorFlow, Keras, and other necessary libraries (installed via `pip`)

## Project Setup

### Install Python

1. Download and install Python from [Python.org](https://www.python.org/downloads/).
2. Check the box for **"Add Python to PATH"** during installation.

### Install VS Code

1. Download and install [Visual Studio Code](https://code.visualstudio.com/).

### Install Python Extension for VS Code

1. Open VS Code and go to Extensions (`Ctrl+Shift+X`).
2. Search for the **Python** extension by Microsoft and install it.

## Creating a Virtual Environment

1. Open VS Code and create a new folder for the project.
2. Open the terminal (`Ctrl + \`).
3. Run the following command to create a virtual environment:
   ```bash
   python -m venv env
   ```
4. Activate the virtual environment:
   ```bash
   .\env\Scripts\activate  # Windows
   ```
   You should see `(env)` in your terminal prompt.

## Installing Required Libraries

With the virtual environment active, install the required libraries:

```bash
pip install tensorflow keras opencv-python pandas matplotlib seaborn scikit-learn
```

## Dataset

### Skin Cancer Dataset

- Download the **ISIC Skin Cancer Dataset** from [Kaggle](https://www.kaggle.com/) or [ISIC Archive](https://www.isic-archive.com/).
- Organize it as follows:

```
dataset/
    melanoma/
        image1.jpg
        image2.jpg
        ...
    nevus/
        image1.jpg
        image2.jpg
        ...
    basal_cell_carcinoma/
        image1.jpg
        image2.jpg
        ...
    squamous_cell_carcinoma/
        image1.jpg
        image2.jpg
        ...
    vascular_lesion/
        image1.jpg
        image2.jpg
        ...
    seborrheic_keratosis/
        image1.jpg
        image2.jpg
        ...
    pigmented_benign_keratosis/
        image1.jpg
        image2.jpg
        ...
    dermatofibroma/
        image1.jpg
        image2.jpg
        ...
    actinic_keratosis/
        image1.jpg
        image2.jpg
        ...
```

## Data Preprocessing

To load, preprocess, and augment the images:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,  # 80-20 train-validation split
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

train_data = datagen.flow_from_directory(
    'path_to_dataset',  # Replace with your dataset path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    subset='training'
)

validation_data = datagen.flow_from_directory(
    'path_to_dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```

## Building the CNN Model

This is a multi-class CNN architecture that classifies skin cancer into different categories:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(9, activation='softmax'))  # Softmax for multi-class classification

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

## Training the Model

Train the model:

```python
history = model.fit(
    train_data,
    epochs=20,
    validation_data=validation_data
)
```

## Evaluating the Model

Evaluate the model on the validation set:

```python
loss, accuracy = model.evaluate(validation_data)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
```

## Saving and Loading the Model

To save the trained model:

```python
model.save('skin_cancer_classification_model.h5')
```

To load it for future use:

```python
from tensorflow.keras.models import load_model
model = load_model('skin_cancer_classification_model.h5')
```

## Making Predictions

To make predictions on new images:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img('path_to_new_image', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
class_labels = ['melanoma', 'nevus', 'basal_cell_carcinoma', 'squamous_cell_carcinoma', 
                'vascular_lesion', 'seborrheic_keratosis', 'pigmented_benign_keratosis', 
                'dermatofibroma', 'actinic_keratosis']

print(f'Predicted class: {class_labels[predicted_class]}')
```

## Running the Code

1. Ensure the virtual environment is activated (`.\env\Scripts\activate`).
2. Run the Python script:
   ```bash
   python main.py
   ```

## Optional: Deployment

To deploy the trained model as a web app using Flask or FastAPI, refer to tutorials on deploying TensorFlow models with these frameworks.

---

This version is adjusted to handle multi-class classification using categorical crossentropy.