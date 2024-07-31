import cv2
import pytesseract
from pytesseract import Output
import numpy as np


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def ocr_image(image, to_data=False):
    gray = get_grayscale(image)
    if to_data:
        return pytesseract.image_to_data(gray,output_type=Output.DICT, config="--psm 3 --oem 1")
    else:
        return  pytesseract.image_to_string(gray, config="--psm 3 --oem 1")


# # Load an image
# image = cv2.imread("truck.jpg")

# # Convert the image to gray scale
# gray = get_grayscale(image)
# thresh = thresholding(gray)
# openings = opening(gray)
# cannys = canny(gray)
# cv2.imshow("canny", openings)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Perform OCR on the gray scale image
# text = pytesseract.image_to_string(cannys, config="--psm 3 --oem 1")

# # Print the detected text
# print(text)
