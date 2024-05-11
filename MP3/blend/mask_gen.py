import cv2 
import os 
import numpy as np
img_path = "burger.jpg"

# STEP1: Read the image
img1 = cv2.imread("cpu.jpg")
h, w, _ = img1.shape

# Step2: the region of mask according to observation
cpu_width_range = [194, 800]
cpu_height_range = [190, 700]

def resize_img2(img_path):
    img2 = cv2.imread(img_path)
    img2 = cv2.resize(img2, (w, h))
    cv2.imwrite(os.path.join(os.getcwd(), "{}_resize.{}".format(\
        img_path.split(".")[0], img_path.split(".")[1])), img2)

resize_img2(img_path)

# Step3: make and save the mask
mask = np.zeros((h, w, 3), dtype=np.uint8)
mask[cpu_height_range[0]:cpu_height_range[1], cpu_width_range[0]:cpu_width_range[1]] = 255
cv2.imwrite("mask_insterst.png", mask)