import numpy as np
import scipy
import cv2
k = 0.035
T = 0.999
sigma = 3
gaussian_windows_size = 9
nms_windows = 9

def calculate_harrris_mat(dx, dy):
# Get the harris matrix element for every element 
  fxx = dx * dx 
  fyy = dy * dy 
  fxy = dx * dy 
  
  M_xx = cv2.GaussianBlur(fxx, (gaussian_windows_size, gaussian_windows_size), sigma)
  M_yy = cv2.GaussianBlur(fyy, (gaussian_windows_size, gaussian_windows_size), sigma)
  M_xy = cv2.GaussianBlur(fxy, (gaussian_windows_size, gaussian_windows_size), sigma)
  return M_xx, M_yy, M_xy

def nms(corners, response, window_size):
  corners_new = np.zeros_like(corners)
  for i in range(0, corners.shape[0]):
    for j in range(0, corners.shape[1]):
      max_windows = corners[max(0, i - window_size // 2) : min(i + window_size // 2 + 1, response.shape[0]),\
                            max(0, j - window_size // 2) : min(j + window_size // 2 + 1, response.shape[1])]
      if np.max(max_windows) > corners[i, j]:
        corners_new[i, j] = 0
      else:
        corners_new[i, j] = corners[i, j]
  return corners_new
def expand(response, window_size):
  for i in range(response.shape[0]):
    for j in range(response.shape[1]):
      max_windows = response[max(0, i - window_size // 2) : min(i + window_size // 2 + 1, response.shape[0]),\
                            max(0, j - window_size // 2) : min(j + window_size // 2 + 1, response.shape[1])]
      max_value = np.max(max_windows)
      if max_value < 0:  
        response[i, j] = 0
  return response
def compute_corners(I):
  # Currently this code proudces a dummy corners and a dummy corner response
  # map, just to illustrate how the code works. Your task will be to fill this
  # in with code that actually implements the Harris corner detector. You
  # should return th ecorner response map, and the non-max suppressed corners.
  # Input:
  #   I: input image, H x W x 3 BGR image
  # Output:
  #   response: H x W response map in uint8 format
  #   corners: H x W map in uint8 format _after_ non-max suppression. Each
  #   pixel stores the score for being a corner. Non-max suppressed pixels
  #   should have a low / zero-score.
  # STEP0: convert the image to grayscale
  I = 0.114 * I[:, :, 0] + 0.587 * I[:, :, 1] + 0.299 * I[:, :, 2]
  
  # dif_filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
  dif_filter = [[-1., 0, 1.]]
  # STEP1: compute image gradients 
  dx = scipy.signal.convolve2d(I, np.array(dif_filter), mode='same', boundary='symm')
  dy = scipy.signal.convolve2d(I, np.array(dif_filter).T, mode='same', boundary='symm')
  # STEP2: compute the harris matrix
  M_xx, M_yy, M_xy = calculate_harrris_mat(dx, dy)
  # STEP3: calculate response function
  det_M = M_xx * M_yy - M_xy * M_xy
  trace_M = M_xx + M_yy 
  # response = det_M - k * (trace_M ** 2)
  response = trace_M + np.sqrt(trace_M ** 2 - 4 * det_M)
  
  # STEP4: threshold response function
  # response = expand(response, nms_windows)
  # response = response / response.max() * 255
  corners = np.where(response > T * np.max(response), response / response.max() * 255, 0)
  
  # STEP5: non-maximum suppression
  # corners = nms(corners, response, nms_windows)
  
  corners = corners.astype(np.uint8)
  response = np.clip(response, 0, 255)
  response = response.astype(np.uint8)
  return response, corners
