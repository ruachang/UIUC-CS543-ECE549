import numpy as np
import scipy 
from scipy.ndimage import gaussian_filter

levels = 10
sigma_windows = 2
sigma_img_stack = 2

def colve_mult_channel(img, filter, mode, boundary):
  img_channel_stack = [scipy.signal.convolve2d(img[:, :, i], filter, mode=mode, boundary=boundary) for i in range(3)]
  return np.stack(img_channel_stack, axis=-1)

def gaussian_filter1(sigma, size):
# Only smooth the image
  gaussian_filter = np.zeros((size, size)) 
  max_x = size // 2
  for i in range(-max_x, max_x+1):
    for j in range(-max_x, max_x+1):
      gaussian_filter[i + max_x][j + max_x] = (1 / (sigma ** 2)) * np.exp(-((i ** 2 + j ** 2) / (2 * (sigma ** 2))))
  sum_filter = np.sum(gaussian_filter)
  gaussian_filter = gaussian_filter / sum_filter
  return gaussian_filter

def gaussian_img_stack(img, sigma, num_levels):
  img_stack = [img]
  dog_stack = []
  current_sigma = sigma
  for l in range(num_levels):
    # gaussian_windows = gaussian_filter1(current_sigma, 2 * windows_size + 1)
    # gaussian_img = scipy.signal.convolve2d(img_stack[l], gaussian_windows, mode="same", boundary="symm")
    gaussian_img = gaussian_filter(img_stack[l], current_sigma)
    
    dog = img_stack[l] - gaussian_img
    img_stack.append(gaussian_img)
    dog_stack.append(dog)
    current_sigma = current_sigma * np.sqrt(2)
  dog_stack.append(gaussian_img)
  return img_stack, dog_stack

def normalize(img):
  min_val = np.min(img)
  max_val = np.max(img)
  img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
  
  return img

def blend(im1, im2, mask):
  # im1 = 0.114 * im1[:, :, 0] + 0.587 * im1[:, :, 1] + 0.299 * im1[:, :, 2]
  # im2 = 0.114 * im2[:, :, 0] + 0.587 * im2[:, :, 1] + 0.299 * im2[:, :, 2]
  # mask = 0.114 * mask[:, :, 0] + 0.587 * mask[:, :, 1] + 0.299 * mask[:, :, 2]
  im1 = np.array(im1, dtype=np.float32)
  im2 = np.array(im2, dtype=np.float32)
  mask = np.array(mask, dtype=np.float32)
  img1_gaussian_stack, img1_dog_stack = gaussian_img_stack(im1, sigma_img_stack, levels)
  img2_gaussian_stack, img2_dog_stack = gaussian_img_stack(im2, sigma_img_stack, levels)
  mask_gaussian, _ = gaussian_img_stack(mask, sigma_windows, levels)
  combined_gaussian = []
  out = np.zeros_like(mask_gaussian[-1])
  for l in range(levels):
    mask_gaussian[l] = mask_gaussian[l] / 255.
    combined_img = mask_gaussian[l] * img1_dog_stack[l] + (1 - mask_gaussian[l]) * img2_dog_stack[l]
    out += combined_img
  #   combined_gaussian.append(combined_img)
  # for l in range(levels):
  #   out += combined_gaussian[l]
  out = normalize(out)
  return out
