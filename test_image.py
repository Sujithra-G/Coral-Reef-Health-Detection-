import tensorflow as tf
import numpy as np
import cv2
import os

model = tf.keras.models.load_model("coral_cnn_model.h5")

img_path = input("Enter image path: ")

if not os.path.exists(img_path):
    print("Error: Image not found!")
    exit()

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = cv2.GaussianBlur(img, (5,5), 0)

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img

img = preprocess(img_path)

prediction = model.predict(img)

classes = ["Bleached", "Dead", "Healthy"]

print("Prediction:", classes[np.argmax(prediction)])