# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:05:49 2026

@author: Richard
"""

import pytesseract
import cv2
import re


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# load image
img = cv2.imread("Ca.png")

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold improves numeric OCR
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# run OCR
text = pytesseract.image_to_string(thresh, config="--psm 6")
text = text.replace("O", "0")
text = text.replace("l", "1")
print(text)
