import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array

# Load Model
model = keras.models.load_model('./models/happy_or_sad.keras')

# Create directory to save images
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Continuously capture camera input
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert BGR frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert image to grayscale and resize
    img_pil = Image.fromarray(frame_rgb)
    img_pil_gray = img_pil.convert('L')
    img_pil_gray = img_pil_gray.resize((48, 48))

    # Save grayscale image
    img_gray_array = np.array(img_pil_gray)

    # Convert PIL Image to numpy array
    input_data = img_to_array(img_pil_gray)
    input_data = np.expand_dims(input_data, axis=0)

    # Perform prediction
    prediction = model.predict(input_data, verbose=0)
    y = np.argmax(prediction)

    # Set label based on prediction
    if (y == 0):
        label = "Happy"
    elif (y == 1):
        label = "Sad"

    # Resize the transformed grayscale image to maintain aspect ratio
    height, width = frame_rgb.shape[:2]
    img_transformed_gray = cv2.resize(img_gray_array, (int(width * 0.2), int(height * 0.2)))

    # Convert grayscale image to BGR for display
    img_transformed_gray_bgr = cv2.cvtColor(img_transformed_gray, cv2.COLOR_GRAY2BGR)
    img_transformed_gray_bgr = cv2.flip(img_transformed_gray_bgr, 1)
    
    # Display the label/title on the left side of the screen
    cv2.putText(img_transformed_gray_bgr, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with the label/title
    cv2.imshow('Image', img_transformed_gray_bgr)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
