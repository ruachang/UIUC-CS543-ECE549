import numpy as np
import scipy

k = 0.11
T = 0.01
sigma = 1
gaussian_windows_size = 3
nms_windows = 3
def colve_mult_channel(img, filter, mode, boundary):
  img_channel_stack = [scipy.signal.convolve2d(img[:, :, i], filter, mode=mode, boundary=boundary) for i in range(3)]
  return np.stack(img_channel_stack, axis=-1)

def gaussian_filter(sigma, size):
# Only smooth the image
  gaussian_filter = np.zeros((size, size)) 
  max_x = size // 2
  for i in range(-max_x, max_x+1):
    for j in range(-max_x, max_x+1):
      gaussian_filter[i + max_x][j + max_x] = (1 / (sigma ** 2)) * np.exp(-((i ** 2 + j ** 2) / (2 * (sigma ** 2))))
  sum_filter = np.sum(gaussian_filter)
  gaussian_filter = gaussian_filter / sum_filter
  return gaussian_filter

def calculate_harrris_mat(dx, dy, gaussian_filter):
# Get the harris matrix element for every element 
  fxx = dx * dx 
  fyy = dy * dy 
  fxy = dx * dy 
  
  # M_xx = scipy.ndimage.gaussian_filter(fxx, sigma)
  # M_yy = scipy.ndimage.gaussian_filter(fyy, sigma)
  # M_xy = scipy.ndimage.gaussian_filter(fxy, sigma)
  M_xx = scipy.signal.convolve2d(fxx, gaussian_filter, mode='same', boundary='symm')
  M_yy = scipy.signal.convolve2d(fyy, gaussian_filter, mode='same', boundary='symm')
  M_xy = scipy.signal.convolve2d(fxy, gaussian_filter, mode='same', boundary='symm')
  
  return M_xx, M_yy, M_xy

def nms(corners, response, window_size):
  for i in range(window_size // 2, corners.shape[0] - window_size // 2):
    for j in range(window_size // 2, corners.shape[1] - window_size // 2):
      max_windows = response[i - window_size // 2 : i + window_size // 2 + 1,\
                            j - window_size // 2 : j + window_size // 2 + 1]
      if np.max(max_windows) > response[i, j]:
        corners[i, j] = 0
  return corners
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
  dif_filter = [[-1, 0, 1]]#, [0, 0, 0], [1, 2, 1]]
  # STEP1: compute image gradients 
  dx = scipy.signal.convolve2d(I, np.array(dif_filter), mode='same', boundary='symm')
  dy = scipy.signal.convolve2d(I, np.array(dif_filter).T, mode='same', boundary='symm')
  # STEP2: compute the harris matrix
  gaussian_window = gaussian_filter(sigma, gaussian_windows_size)
  M_xx, M_yy, M_xy = calculate_harrris_mat(dx, dy, gaussian_window)
  # STEP3: calculate response function
  det_M = M_xx * M_yy - M_xy * M_xy
  trace_M = M_xx + M_yy 
  response = det_M - k * (trace_M ** 2)
  # response = det_M / (trace_M ** 2 + 1e-10)
  # STEP4: threshold response function
  corners = np.where(response > T * np.max(response), response, 0)
  # STEP5: non-maximum suppression
  corners = nms(corners, response, nms_windows)
  corners = corners.astype(np.uint8)
  response = response.astype(np.uint8)
  return response, corners
