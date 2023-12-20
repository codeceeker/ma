import pytesseract
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Set the path to the Tesseract executable (modify this according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    # Load image using Pillow (PIL)
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels (for simplicity)
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Normalize pixel values to be between 0 and 1
    img_array = img_array / 255.0
    
    # Expand dimensions to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to create a simple neural network model for digit recognition
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Train the model on a dataset (MNIST for simplicity)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Create and train the neural network model
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Test the model on a sample image (replace 'your_image.png' with the path to your image)
image_path = 'your_image.png'
img_array = load_and_preprocess_image(image_path)

# Make predictions using the trained model
predictions = model.predict(img_array)
predicted_label = np.argmax(predictions)

# Use Tesseract OCR to extract text from the image
extracted_text = pytesseract.image_to_string(Image.open(image_path))

# Display the results
print("Predicted Digit:", predicted_label)
print("Extracted Text:", extracted_text)
