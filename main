from data_preprocessing import load_data
from model import build_model
import tensorflow as tf

# Load data
train_data, validation_data = load_data('dataset/')

# Build and compile the model
model = build_model()

# Train the model
history = model.fit(train_data, epochs=20, validation_data=validation_data)

# Save the model
model.save('skin_cancer_classification_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(validation_data)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
