import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/happy_or_sad.keras")

model = load_model(MODEL_PATH)

ARCHITECTURE_IMAGE_PATH = os.path.join(BASE_DIR, "happy_or_sad_model.png")
plot_model(model, to_file=ARCHITECTURE_IMAGE_PATH, show_shapes=True, show_layer_names=True)
print(f"Model architecture saved as {ARCHITECTURE_IMAGE_PATH}")