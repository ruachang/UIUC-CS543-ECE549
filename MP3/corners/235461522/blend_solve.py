import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

def Laplacian_stack(img, sigmas):
    current_img = img
    stack = []
    for sigma in sigmas:
        blurred = gaussian_filter(current_img, sigma)
        stack.append(current_img - blurred)
        current_img = blurred
    stack.append(current_img)
    return stack

def Gaussian_stack(mask, sigmas):
    stack = [mask]
    new_mask = mask
    for sigma in sigmas:
        new_mask = gaussian_filter(new_mask, sigma)
        stack.append(new_mask)
    return stack

def combine(im1, im2, GR):
    GR = GR / 255.
    return im1 * GR + (1 - GR) * im2

def blend(im1, im2, mask):
    im1 = np.mean(im1, axis=2)
    im2 = np.mean(im2, axis=2)
    mask = np.mean(mask, axis=2)
    sigmas = [2*i for i in range(10)]
    LA = Laplacian_stack(im1, sigmas)
    LB = Laplacian_stack(im2, sigmas)
    GR = Gaussian_stack(mask, sigmas)
    blended = np.zeros(LA[-1].shape)
    for i in range(len(LA)):
        blended += combine(LA[i], LB[i], GR[i])
    return blended