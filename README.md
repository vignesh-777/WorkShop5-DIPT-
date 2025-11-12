# License Plate Detection using OpenCV and Haar Cascade

### Author: Vignesh R
### Date: November 12, 2025
### Course: Computer Vision / AI Applications
### Assignment: Workshop 5 — License Plate Detection using Haar Cascade

# Project Overview

This project implements a License Plate Detection system using OpenCV’s Haar Cascade Classifier.

The model identifies and locates vehicle license plates in an input image, draws bounding boxes, and extracts (crops) the plate region for further analysis.

The Haar Cascade used is haarcascade_russian_plate_number.xml — a pre-trained classifier provided by OpenCV.

# Algorithm
1. Read the input image containing the vehicle using OpenCV.

2. Convert the image to grayscale to simplify processing.

3. Load the Haar Cascade classifier for license plate detection.

4. Apply the classifier using detectMultiScale() to locate plate regions.

5. Draw bounding boxes around the detected license plates.

6. Crop and save the detected plate area as a separate image for further use.

# Program
```

# Step 1: Import Libraries
import cv2
import os
import urllib.request
import matplotlib.pyplot as plt

# Replace 'car.jpg' with your test image filename
img = cv2.imread('car.jpg')

# Convert image from BGR to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Input Image")
plt.axis('off')
plt.show()


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml"
cascade_file = "haarcascade_russian_plate_number.xml"

if not os.path.exists(cascade_file):
    urllib.request.urlretrieve(cascade_url, cascade_file)
    print(" Haar Cascade downloaded successfully.")
else:
    print(" Haar Cascade file found.")

plate_cascade = cv2.CascadeClassifier(cascade_file)

if plate_cascade.empty():
    raise IOError(" Haar Cascade failed to load. Check file path or download again.")
else:
    print(" Haar Cascade loaded successfully.")

# Apply Gaussian blur and histogram equalization to improve detection
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
gray_eq = cv2.equalizeHist(gray_blur)

plt.imshow(gray_eq, cmap='gray')
plt.title("Preprocessed Image")
plt.axis('off')
plt.show()

plates = plate_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=5)

print(f"Detected {len(plates)} plate(s).")

# Draw bounding boxes on a copy of the original image
output_img = img_rgb.copy()

for (x, y, w, h) in plates:
    cv2.rectangle(output_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    plate_region = img[y:y+h, x:x+w]
    cv2.imwrite(f"plate_{x}_{y}.png", plate_region)  # Save cropped plate image

plt.imshow(output_img)
plt.title("Detected License Plate(s)")
plt.axis('off')
plt.show()


```

# Output
<img width="748" height="577" alt="image" src="https://github.com/user-attachments/assets/c8bb358e-0663-46fd-9c52-1d8011761235" />

<img width="784" height="554" alt="image" src="https://github.com/user-attachments/assets/c31348f9-f723-41a5-bd0f-331fa2445c70" />

<img width="642" height="41" alt="image" src="https://github.com/user-attachments/assets/47167ac1-a449-4c46-acec-2d0ce0f1cec0" />

<img width="891" height="575" alt="image" src="https://github.com/user-attachments/assets/f4cbe87d-b2c3-434d-8470-b3e05983f49d" />

<img width="821" height="525" alt="image" src="https://github.com/user-attachments/assets/e8763936-ed46-4e34-8b9d-2b073f29a236" /> 

<img width="1104" height="406" alt="image" src="https://github.com/user-attachments/assets/50232e35-be06-4735-b591-b9f488dee71e" />

# Result

The Haar Cascade classifier successfully detected the license plate region from the input image.
After preprocessing (Gaussian Blur and Histogram Equalization), the detection became more stable and accurate.

