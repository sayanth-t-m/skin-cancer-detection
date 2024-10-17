# skin-cancer-detection
This project is a skin cancer detection model using Convolutional Neural Networks (CNNs) built in Python with TensorFlow and Keras. The model classifies skin cancer into benign or malignant categories using image data.

# Skin Cancer Detection Model Documentation

This project is a skin cancer detection model using Convolutional Neural Networks (CNNs) built in Python with TensorFlow and Keras. The model classifies skin cancer into benign or malignant categories using image data.

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

## Prerequisites
Before starting this project, make sure you have the following installed:
- Python 3.7+
- Visual Studio Code
- TensorFlow, Keras, and other necessary libraries (installed via pip)

## Project Setup

### Install Python
1. Download and install Python from the official website [Python.org](https://www.python.org/downloads/).
2. During installation, check the box for **"Add Python to PATH"**.

### Install VS Code
1. Download and install [Visual Studio Code](https://code.visualstudio.com/).

### Install Python Extension for VS Code
1. Open VS Code and go to the Extensions view (`Ctrl+Shift+X`).
2. Search for the **Python** extension by Microsoft and install it.

## Creating a Virtual Environment
A virtual environment ensures dependencies remain isolated and prevents conflicts between projects.

1. Open VS Code and create a new folder for the project.
2. Open the terminal (`Ctrl + ` ``).
3. Run the following command to create a virtual environment:
   ```bash
   python -m venv env
   ```
4. To activate the virtual environment, run:
   ```bash
   .\env\Scripts\activate
   ```
   You should see `(env)` in your terminal prompt.

## Installing Required Libraries
With the virtual environment active, install the following required libraries:

```bash
pip install tensorflow keras opencv-python pandas matplotlib seaborn scikit-learn
```

## Dataset
### Skin Cancer Dataset
- Download the **ISIC Skin Cancer Dataset** from [Kaggle](https://www.kaggle.com/) or [ISIC Archive](https://www.isic-archive.com/).
- Extract the dataset and organize it as follows:

```
dataset/
    benign/
        image1.jpg
        image2.jpg
        ...
    malignant/
        image1.jpg
        image2.jpg
        ...
```

## Data Preprocessing

To load, preprocess, and augment the images, use the following code:

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
    'path_to_dataset',  # Replace with the path to your dataset
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_data = datagen.flow_from_directory(
    'path_to_dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
```

## Building the CNN Model

To build the CNN architecture, use the following code:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

## Training the Model

Train the model using the following code:

```python
history = model.fit(
    train_data,
    epochs=20,
    validation_data=validation_data
)
```

## Evaluating the Model

Evaluate your model on the validation dataset:

```python
loss, accuracy = model.evaluate(validation_data)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
```

## Saving and Loading the Model

To save the trained model:

```python
model.save('skin_cancer_detection_model.h5')
```

To load the saved model for future use:

```python
from tensorflow.keras.models import load_model
model = load_model('skin_cancer_detection_model.h5')
```

## Making Predictions

You can make predictions on new images as follows:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img('path_to_new_image', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
if prediction > 0.5:
    print("Malignant")
else:
    print("Benign")
```

## Running the Code

1. Ensure the virtual environment is activated (`.\env\Scriptsctivate`).
2. Run the Python script:
   ```bash
   python main.py
   ```

## Optional: Deployment

To deploy the trained model as a web app using Flask or FastAPI, refer to tutorials on deploying TensorFlow models with these frameworks.
