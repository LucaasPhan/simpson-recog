from train import model, IMG_SIZE, characters
import cv2 as cv
import matplotlib.pyplot as plt
import caer
import numpy as np

test_path = r'simpson-recog/Bart_Simpson.png'
img = cv.imread(test_path)

def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMG_SIZE)
    img = caer.reshape(img, IMG_SIZE, 1)
    return img

predictions = model.predict(prepare(img))
print(characters[np.argmax(predictions[0])])
