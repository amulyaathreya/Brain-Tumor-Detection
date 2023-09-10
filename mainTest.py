import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10EpochsCategorical.keras')

image = cv2.imread('C:\\Users\\AMULYA ATHREYA\\Downloads\\archive\\pred\\pred0.jpg')

img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

input_img = np.expand_dims(img, axis=0)

# Predict probabilities
probabilities = model.predict(input_img)

# Get the class with the highest probability
predicted_class = np.argmax(probabilities[0])

if predicted_class == 0:
    print("No Tumor")
else:
    print("Tumor")

