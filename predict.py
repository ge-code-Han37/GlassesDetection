import cv2 as cv
import os
import numpy as np
import pickle
from skimage.feature import hog

IMG_SIZE=64

CATEGORIES =['glasses', 'noGlasses']

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'glasses_model.pkl')

with open(MODEL_PATH ,'rb') as file:
    model=pickle.load(file)


def predict(img):

    img =cv.resize(img,(IMG_SIZE,IMG_SIZE))
    img =cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    features = features.reshape(1,-1)


    prediction = model.predict(features)[0]

    confidence = model.predict_proba(features)[0] [prediction]
    label = CATEGORIES[prediction]

    return label, confidence



def test_image(image_path):
    img = cv.imread(image_path)

    if img is None:
        print("image not found, check the path")
        return
    
    label, confidence = predict(img)

    text = f"{label} ({confidence*100:.1f}%)"
    color = (0, 255, 0) if label == 'glasses' else (0, 0, 255)
    cv.putText(img, text, (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv.imshow('Test Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()



test_image('toTestMyself/pic6.jpg')






"""

webcam= cv.VideoCapture(0)

print("press q to quit")

while webcam.isOpened():
    isTrue , frame =webcam.read()
    label, confidence = predict(frame)
    text = f"{label} ({confidence*100: .1f}%)"
    color = (0, 255, 0) if label == 'glasses' else (0, 0, 255)
    cv.putText(frame, text ,(30,50) , cv.FONT_HERSHEY_TRIPLEX , 1.2 , color , 3 )
    cv.imshow('Glasses Detector', frame)

    if cv.waitKey(10) == ord('q'):
        break



webcam.release()
cv.destroyAllWindows()



"""

