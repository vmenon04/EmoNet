import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Function to get all filepaths in a directory
def get_filepaths(directory):
    filepaths = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            filepaths.append(filepath)
    return filepaths

# Load Model
model = keras.models.load_model('./models/happy_or_sad.keras')

# Set the data generator object
datagen_test = ImageDataGenerator(rescale=1./255, zoom_range=0.3, horizontal_flip=True)

# Set the directory for the test data (all happy images)
filepaths = get_filepaths('./data/happy_or_sad/validation/happy/')

counter = 0
total = len(filepaths)

# Loop through all the images in the test data
for i in tqdm(range(len(filepaths))):
    file = filepaths[i]
    try:
        # Load the image
        image = keras.preprocessing.image.load_img(file, target_size=(48, 48), color_mode='grayscale')

        # Convert the image to a numpy array
        input_data = keras.preprocessing.image.img_to_array(image)

        # Expand the dimensions of the image
        input_data = np.expand_dims(input_data, axis=0)

        # Perform prediction
        prediction = model.predict(input_data, verbose=0)

        # Get the index of the highest probability (0 or 1)
        y = np.argmax(prediction)

        # If the prediction is correct, increment the counter
        if (y==0):
            counter+=1

    # If there is an error, skip the image
    except:
        total-=1
        continue

# Print the accuracy for happy images
print("Happy Accuracy: " + str(counter) + "/" + str(total) + " = " + str(counter/total))


# Set the directory for the test data (all sad images)
filepaths = get_filepaths('./data/happy_or_sad/validation/sad/')

counter = 0
total = len(filepaths)

# Loop through all the images in the test data
for i in tqdm(range(len(filepaths))):
    file = filepaths[i]
    try:
        # Load the image and perform the same steps as before, but for sad images
        image = keras.preprocessing.image.load_img(file, target_size=(48, 48), color_mode='grayscale')
        input_data = keras.preprocessing.image.img_to_array(image)
        input_data = np.expand_dims(input_data, axis=0)
        prediction = model.predict(input_data, verbose=0)
        y = np.argmax(prediction)
        if (y==1):
            counter+=1
    
    # Skip the image if there is an error
    except:
        total-=1
        continue

# Print the accuracy for sad images
print("Sad Accuracy: " + str(counter) + "/" + str(total) + " = " + str(counter/total))