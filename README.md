# Glasses Detection

A computer vision project that detects whether a person is wearing glasses or not, using HOG feature extraction and an SVM classifier.

Built as part of my computer vision learning path — no deep learning, just classical ML and signal processing.

---

## How it works

1. Images are resized to 64x64 and converted to grayscale
2. HOG (Histogram of Oriented Gradients) features are extracted — captures edges and shapes rather than raw pixel values
3. An SVM classifier is trained on those features
4. The trained model is saved and used for prediction on new images or live webcam feed

---

## Results

- ~79% accuracy on held-out test data
- Trained on 130 images (65 glasses / 65 no glasses)
- Balanced dataset, evaluated with precision, recall, and f1-score

---

## Project Structure

```
GLASSESDETECTION/
├── glasses/          # training images — with glasses
├── noGlasses/        # training images — without glasses
├── ToTestByMyself/   # personal test images (unseen during training)
├── train.py          # loads data, extracts HOG features, trains and saves SVM model
├── predict.py        # loads saved model, runs prediction on image file or webcam
└── README.md
```

---

## Requirements

```bash
pip install opencv-python scikit-learn scikit-image numpy
```

---

## Usage

**Train the model:**
```bash
python train.py
```

**Test on an image:**

In `predict.py`, call:
```python
test_image('ToTestByMyself/yourimage.jpg')
```

**Run on webcam:**

Uncomment the webcam block in `predict.py` and run:
```bash
python predict.py
```

---

## What I'd improve with more time

- Larger and more diverse dataset (200+ per class)
- Face detection step before classification so it works on full photos not just cropped faces
- CNN-based classifier for higher accuracy
