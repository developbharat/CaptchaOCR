import os

import cv2
import numpy as np

def clean_image(image):
    # image = cv2.imread('test.png')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)


    # get image spatial dimensions
    height, width = detected_lines.shape[:2]
    cv2.line(detected_lines, (width, 0), (0, height), (0,0,0), 1)

    # Dilate image
    kernel = np.ones((2, 2), np.uint8)
    dilated_image = cv2.dilate(detected_lines, kernel, iterations=1)

    # Convert image back to rgb
    dilated_image = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2RGB)
    dilated_image = cv2.bitwise_not(dilated_image)
    return dilated_image



basedir = "data/dataset1/train"
files = [os.path.join(basedir, file) for file in os.listdir(basedir) if file.endswith(".png")]

for filepath in files:
    picture = cv2.imread(filepath)
    picture = clean_image(picture)
    cv2.imwrite(filepath, picture)
