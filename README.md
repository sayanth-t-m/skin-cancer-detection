
# Skin Cancer Detection Using CNN

This project implements a Convolutional Neural Network (CNN) to detect different types of skin cancer using images from the [Skin Cancer ISIC](https://www.isic-archive.com) dataset. The model classifies images into 9 categories of malignant and benign oncological diseases.

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Making Predictions](#making-predictions)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [References](#references)

## Project Structure

```
skin-cancer-detection/
│
├── dataset/                  # Folder to store your dataset
│   ├── melanoma/             # Folder for melanoma images
│   ├── nevus/                # Folder for nevus images
│   ├── basal_cell_carcinoma/ # Folder for basal cell carcinoma images
│   ├── squamous_cell_carcinoma/ # Folder for squamous cell carcinoma images
│   ├── vascular_lesion/      # Folder for vascular lesion images
│   ├── seborrheic_keratosis/ # Folder for seborrheic keratosis images
│   ├── pigmented_benign_keratosis/ # Folder for pigmented benign keratosis images
│   ├── dermatofibroma/       # Folder for dermatofibroma images
│   ├── actinic_keratosis/    # Folder for actinic keratosis images
│
├── main.py                   # Main Python script for the model
├── model.py                  # Script to define and build the CNN model
├── data_preprocessing.py      # Script for data preprocessing
└── requirements.txt          # Required libraries
```

## Dataset

The dataset consists of **2357 images** of malignant and benign skin conditions, formed from the **International Skin Imaging Collaboration (ISIC)**. The dataset contains the following skin diseases:

- Actinic Keratosis
- Basal Cell Carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented Benign Keratosis
- Seborrheic Keratosis
- Squamous Cell Carcinoma
- Vascular Lesion

Each category of images is stored in its respective folder inside the `dataset/` directory.

## Data Preprocessing

The images are preprocessed using the `ImageDataGenerator` class from Keras to enhance generalization and normalize pixel values. 

Preprocessing steps include:

- **Rescaling**: Normalize pixel values to the range [0, 1].
- **Augmentation**: Apply random horizontal flipping, rotations, and zooms.
- **Splitting**: 80% of the data is used for training and 20% for validation.

The preprocessing code is located in `data_preprocessing.py`:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(dataset_path):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.2
    )

    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    return train_data, validation_data
```

## Model Architecture

The CNN model consists of several convolutional layers, followed by max-pooling and fully connected (dense) layers. Dropout is used to prevent overfitting.

The model architecture is defined in `model.py`:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model():
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
    model.add(Dense(9, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

## Training the Model

To train the model, run the `main.py` script. The training will last for 20 epochs, and the dataset is split into 80% training and 20% validation data.

Example code from `main.py`:

```python
from data_preprocessing import load_data
from model import build_model

# Load the data
train_data, validation_data = load_data('dataset/')

# Build the model
model = build_model()

# Train the model
history = model.fit(train_data, epochs=20, validation_data=validation_data)
```

## Evaluating the Model

Once the model has been trained, you can evaluate it on the validation set:

```python
loss, accuracy = model.evaluate(validation_data)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
```

## Making Predictions

To make predictions on new images, use the following code:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess the image
img = image.load_img('path_to_new_image', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make the prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Define the class labels
class_labels = ['melanoma', 'nevus', 'basal_cell_carcinoma', 'squamous_cell_carcinoma', 
                'vascular_lesion', 'seborrheic_keratosis', 'pigmented_benign_keratosis', 
                'dermatofibroma', 'actinic_keratosis']

# Output the predicted class
print(f'Predicted class: {class_labels[predicted_class]}')
```

## Requirements

To install the required libraries, run:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes the following libraries:
- `tensorflow`
- `numpy`
- `Pillow`

## How to Run

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your_username/skin-cancer-detection.git
    cd skin-cancer-detection
    ```

2. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Organize your dataset** by placing the images in the correct folders under the `dataset/` directory.

4. **Run the model**:
    ```bash
    python main.py
    ```

## References

1. [The International Skin Imaging Collaboration (ISIC)](https://www.isic-archive.com)
2. TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
