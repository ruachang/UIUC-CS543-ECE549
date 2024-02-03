import sys
sys.path.append('..')
import os
import imageio
import numpy as np
from absl import flags, app
import copy
FLAGS = flags.FLAGS
import cv2 
flags.DEFINE_string('test_name_noise', 'scan_almastatue', 
                    'what set of shreads to load')


def load_imgs(name):
    file_names = os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shredded-images', name))
    file_names.sort()
    Is = []
    for f in file_names:
        I = imageio.v2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shredded-images', name, f))
        Is.append(I)
    return Is

def normalize(x):
    mean = np.mean(x)
    norm = np.linalg.norm(x - mean) 
    return (x - mean) / norm

def midvalue_filter(kernel_size, image):
    image_filter = copy.deepcopy(image)
    # image_filter = cv2.copyMakeBorder(image_filter, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    height, width, channel = image.shape
    for c in range(channel):
        for i in range(height):
            for j in range(width):
                if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                    continue
                else:
                    # sliced_image = image[i - 1 : i + 2, j - 1 : j + 2, c]
                    image_filter[i, j, c] = np.mean(image[i - 1 : i + 2, j - 1 : j + 2, c])
    return image_filter
def pixel_norm(image_right, image_left):
# Assume that the slide range is depend on the 20% of the right image
    image_right = normalize(image_right)
    # image_right_cut = midvalue_filter(3, image_right[:, 0 : 2, :])
    image_right_cut = image_right[:, 0 : 2, :]
    
    image_left = normalize(image_left)
    # image_left_cut = midvalue_filter(3, image_left[:, -3 : -1, :])
    image_left_cut = image_left[:, -3 : -1, :]
    height_l, width, channel = image_left.shape
    height_r, _, _ = image_right.shape
    max_slide_height = int(height_r * 0.2)
    min_dis = float('inf') 
    min_slide = 0
    for slide_height in range(-max_slide_height, max_slide_height):
        overlap_bottom = min(height_r, height_l + slide_height)  
        overlap_top = max(0, slide_height)
        dis = np.float64(0)
        # calculate the start and end point of the finish height
        overlap_right = np.float64((image_right_cut[overlap_top : overlap_bottom, :, :]))
        overlap_left = np.float64((image_left_cut[-min(0, slide_height) : -min(0, slide_height) + overlap_bottom - overlap_top, :, :]))
        # overlap_left = midvalue_filter(3, overlap_left)
        # overlap_right = normalize(overlap_right)
        # overlap_right = midvalue _filter(3, overlap_right)
        # dis = np.sum((overlap_right - overlap_left) ** 2)
        dis = -np.dot(overlap_right.flatten(), overlap_left.flatten())
        if dis < min_dis:
            min_dis = dis
            min_slide = slide_height
    return min_dis, min_slide
def pairwise_distance(Is):
    '''
    :param Is: list of N images
    :return dist: pairwise distance matrix of N x N
    
    Given a N image stripes in Is, returns a N x N matrix dist which stores the
    distance between pairs of shreds. Specifically, dist[i,j] contains the
    distance when strip j is just to the left of strip i. 
    '''
    dist = np.ones((len(Is), len(Is)))
    slide_dist = np.ones((len(Is), len(Is)))
    # Calculate the pairwise distance between two shred images 
    for i in range(len(Is)):
        for j in range(len(Is)):
            dist[i][j], slide_dist[i][j] = pixel_norm(Is[i], Is[j])
    return dist, slide_dist

def solve(Is):
    '''
    :param Is: list of N images
    :return order: order list of N images
    :return offsets: offset list of N ordered images
    '''
    order = [10, 3, 15, 16, 13, 0, 11, 1, 2, 7, 8, 9, 5, 17, 4, 14, 6, 12]
    offsets = [43, 0, 7, 24, 51, 49, 52, 35, 48, 45, 17, 21, 27, 2, 38, 32, 31, 34]
    # We are returning the order and offsets that will work for 
    # hard_campus, you need to write code that works in general for any given
    # Is. Use the solution for hard_campus to understand the format for
    # what you need to return
    dist, slide_dist = pairwise_distance(Is)

    inds = np.arange(len(Is))
    # run greedy matching
    order = [0]
    offsets_tmp = []
    # Still use the given greedy algorithm 
    # First only store all of the distance from the right shred
    for i in range(len(Is) - 1):
        d1 = np.min(dist[0, 1:])
        d2 = np.min(dist[1:, 0])
        if d1 < d2:
        # the found shred is on the left of the current one
            ind = np.argmin(dist[0, 1:]) + 1
            order.insert(0, inds[ind])
            offsets_tmp.insert(0, slide_dist[0, ind])
            dist[0, :] = dist[ind, :]
            slide_dist[0, :] = slide_dist[ind, :]
            dist = np.delete(dist, ind, 0)
            dist = np.delete(dist, ind, 1)
            slide_dist = np.delete(slide_dist, ind, 0)
            slide_dist = np.delete(slide_dist, ind, 1)
            inds = np.delete(inds, ind, 0)
        else:
        # the found shred is on the right of the current one
            ind = np.argmin(dist[1:, 0]) + 1
            order.append(inds[ind])
            offsets_tmp.append(slide_dist[ind, 0])
            dist[:, 0] = dist[:, ind]
            slide_dist[:, 0] = slide_dist[:, ind]
            dist = np.delete(dist, ind, 0)
            dist = np.delete(dist, ind, 1)
            slide_dist = np.delete(slide_dist, ind, 0)
            slide_dist = np.delete(slide_dist, ind, 1)
            inds = np.delete(inds, ind, 0)
    offset = [0]
    # Calculate the offset from the canvas
    for i in offsets_tmp:
        offset.append(offset[-1] + i)
    offset = [int(-(i - max(offset))) for i in offset]
    return order, np.array(offset)


def composite(Is, order, offsets):
    Is = [Is[o] for o in order]
    strip_width = 1
    W = np.sum([I.shape[1] for I in Is]) + len(Is) * strip_width
    H = np.max([I.shape[0] + o for I, o in zip(Is, offsets)])
    H = int(H)
    W = int(W)
    I_out = np.ones((H, W, 3), dtype=np.uint8) * 255
    w = 0
    for I, o in zip(Is, offsets):
        I_out[o:o + I.shape[0], w:w + I.shape[1], :] = I
        w = w + I.shape[1] + strip_width
    return I_out

def main(_):
    Is = load_imgs(FLAGS.test_name_noise)
    order, offsets = solve(Is)
    I = composite(Is, order, offsets)
    import matplotlib.pyplot as plt
    plt.imshow(I)
    plt.show()

if __name__ == '__main__':
    app.run(main)
