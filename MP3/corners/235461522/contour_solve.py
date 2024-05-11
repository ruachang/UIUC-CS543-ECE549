import numpy as np
from scipy import signal
import cv2

gaussian_windows = 5
sigma = 1.4
threshold_low = 0.125
threshold_high = 0.3

def gaussian_filter(img, sigma, size):
# Only smooth the image
  gaussian_filter = np.zeros((size, size)) 
  max_x = size // 2
  for i in range(-max_x, max_x+1):
    for j in range(-max_x, max_x+1):
      gaussian_filter[i + max_x][j + max_x] = (1 / (sigma ** 2)) * np.exp(-((i ** 2 + j ** 2) / (2 * (sigma ** 2))))
  sum_filter = np.sum(gaussian_filter)
  gaussian_filter = gaussian_filter / sum_filter
  return signal.convolve2d(img, gaussian_filter, mode='same', boundary='symm')

def one_dim_dif_gaussian(sigma, size):
  """
  Use single dimension and one time derivative of Gaussian filter 
  to both smooth the image and calculate for the gradient
  """  
  gaussian_filter = np.zeros(size)
  max_x = size // 2 
  for i in range(-max_x, max_x+1):
    gaussian_filter[i + max_x] = (-i / (sigma ** 2)) * np.exp(-i ** 2 / (2 * (sigma ** 2)))

  return gaussian_filter.reshape((1, size))

def quantize_theta(direction):
# horizontal: (-22.5, 22.5) and (-157.7, 157.5)
# vertical: (67.5, 112.5) (-67.5, -112.5 )
# 45: (112.5, 157.5) (-22.5, -67.5 )
# -45: (22.5, 67.5) (-112.5, -157.5)
  if (-22.5 <= direction and direction < 22.5) or \
    (-180 <= direction and direction <= -157.5) or \
      (157.5 <= direction and direction <= 180):
    return 0
  elif (67.5 <= direction and direction < 112.5) or \
    (-112.5 <= direction and direction < -67.5):
      return 90 
  elif (112.5 <= direction and direction < 157.5) or \
    (-67.5 <= direction and direction < -22.5):
      return 45 
  else:
    return -45

def non_max_suppression(mag, direction):
  # determine the theta of gradient 
  # assume we use 8 connections
  h, w = mag.shape
  for i in range(h):
    for j in range(w):
      theta = quantize_theta(direction[i][j])
      if theta == 0: 
        if i - 1 > 0 and i + 1 < h:
          if mag[i][j] < mag[i - 1][j] or mag[i][j] < mag[i + 1][j]:
            mag[i][j] = 0
        elif i - 1 < 0:
          if mag[i][j] < mag[i + 1][j]:
            mag[i][j] = 0
        else:
          if mag[i][j] < mag[i - 1][j]:
            mag[i][j] = 0
      elif theta == 90:
        if j - 1 > 0 and j + 1 < w:
          if mag[i][j] < mag[i][j - 1] or mag[i][j] < mag[i][j + 1]:
            mag[i][j] = 0
        elif j - 1 < 0:
          if mag[i][j] < mag[i][j + 1]:
            mag[i][j] = 0
        else:
          if mag[i][j] < mag[i][j - 1]:
            mag[i][j] = 0
      elif theta == 45:
        if i + 1 < h and j + 1 < w and i - 1 > 0 and j - 1 > 0:
          if mag[i][j] < mag[i - 1][j + 1] or mag[i][j] < mag[i + 1][j - 1]:
            mag[i][j] = 0
        # southwestern is not reachable
        elif i + 1 > h or j - 1 < 0:
          if i - 1 > 0 and j + 1 < w:
            if mag[i][j] < mag[i - 1][j + 1]:
              mag[i][j] = 0
        # northeastern is not reachable
        elif i - 1 < 0 and j + 1 > w:
          if i + 1 < h and j - 1 > 0:
            if mag[i][j] < mag[i + 1][j - 1]:
              mag[i][j] = 0
      else:
        if i + 1 < h and j + 1 < w and i - 1 > 0 and j + 1 > 0:
          if mag[i][j] < mag[i - 1][j - 1] or mag[i][j] > mag[i + 1][j + 1]:
            mag[i][j] = 0
        # southeastern is not reachable
        elif i + 1 > h or j + 1 > w:
          if i - 1 > 0 and j - 1 > 0:
            if mag[i][j] < mag[i - 1][j - 1]:
              mag[i][j] = 0
        # northwestern is not reachable
        elif i - 1 < 0 or j - 1 < 0:
          if i + 1 < h and j + 1 < w:
            if mag[i][j] < mag[i + 1][j + 1]:
              mag[i][j] = 0
         
  return mag

def threshold_process(img, tl, th):
  """
  Using threshold to process the gradient of the image
  Using double threshold
  """
  max_pixel_value = np.max(img)
  min_pixel_value = np.min(img)
  # setting the threshold according to the max and min of the value
  threshold_low = (tl * (max_pixel_value - min_pixel_value) + min_pixel_value)
  threshold_high = (th * (max_pixel_value - min_pixel_value) + min_pixel_value)
  threshold_low = tl
  
  for i in range(img.shape[0] - 1):
    for j in range(img.shape[1] - 1):
      if img[i][j] < threshold_low: 
        img[i][j] = 0 
      # elif img[i][j] > threshold_high:
      #   img[i][j] = img[i][j]
      #   # img[i][j] = max_pixel_value
      # else: 
      #   if (img[i - 1][j - 1] > threshold_high or img[i - 1][j] > threshold_high or img[i - 1][j + 1] > threshold_high or\
      #     img[i][j - 1] > threshold_high or img[i][j + 1] > threshold_high or \
      #       img[i + 1][j - 1] > threshold_high or img[i + 1][j] > threshold_high or img[i + 1][j + 1] > threshold_high):
      #     img[i][j] = img[i][j]
      #     # img[i][j] = max_pixel_value
      #   else:
      #     img[i][j] = 0
  return img
# def connect_edge(img_low, img_high):
# connect the edge of high threshold using low threshold reference image 
  
def compute_edges_dxdy(I):
  """Returns the norm of dx and dy as the edge response function."""
  I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
  I = I.astype(np.float32)/255.
  
  dif_filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
  I = gaussian_filter(I, sigma, gaussian_windows)
  dx = signal.convolve2d(I, np.array(dif_filter), mode='same', boundary='symm')
  dy = signal.convolve2d(I, np.array(dif_filter).T, mode='same', boundary='symm')
  
  # oned_gaussian_filter = one_dim_dif_gaussian(1, 3)
  # dx = signal.convolve2d(I, np.array(oned_gaussian_filter), mode='same', boundary='symm')
  # dy = signal.convolve2d(I, np.array(oned_gaussian_filter).T, mode='same', boundary='symm')
  mag = np.sqrt(dx**2 + dy**2)
  direction = np.degrees(np.arctan2(dy, dx))
  img = non_max_suppression(mag, direction)
  img = threshold_process(mag, threshold_low, threshold_high)
  img = img / img.max() * 255
  img = np.clip(img, 0, 255)
  img = img.astype(np.uint8)
  
  # img = cv2.Canny(I, 100, 250)
  return img
