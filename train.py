import cv2 as cv
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from skimage.feature import hog

IMG_SIZE = 64

DATA_PATH = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = ['glasses', 'noGlasses']


X, y = [], []

for label, category in enumerate(CATEGORIES):
    folder = os.path.join(DATA_PATH, category)

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        img = cv.imread(filepath)
        if img is None:
            continue

        img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        X.append(features)
        y.append(label)

X= np.array(X)
y= np.array(y)

print(f"Loaded {len(X)} images → glasses={np.sum(y==0)}, noGlasses={np.sum(y==1)}")
print()
print()

X_train, X_test ,y_train , y_test= train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model=SVC(kernel='rbf' , C= 10 ,gamma='scale' ,probability=True)
model.fit(X_train,y_train)

y_pred= model.predict(X_test)
print(classification_report(y_test , y_pred , target_names=CATEGORIES))

with open('glasses_model.pkl', 'wb') as results:
    pickle.dump(model, results)