import cv2
import numpy as np

# the picture directory
pic_dir = "C:\\Users\\User\\Pictures\\eyes.jpg"

cap = cv2.VideoCapture(pic_dir)
# read image
ret, img = cap.read()
if ret:
    # grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert integer to float
    gray1 = np.copy(gray)
    gray1.astype(np.float)
    # normalize gray image
    gray1 = gray1/np.amax(gray1)
    #gamma correction
    gamma = 0.25
    gamma_img = np.power(gray1, gamma) * 255
    gamma_img = gamma_img.astype(np.uint8)
    cv2.imshow("gray image", gray)
    cv2.imshow("gamma corrected image", gamma_img)
    cv2.waitKey(0)
