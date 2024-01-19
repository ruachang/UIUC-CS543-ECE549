import sys
sys.path.append('MP1')
import os
import imageio
import numpy as np
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string('test_name_simple', 'simple_person', 
                    'what set of shreads to load')

def load_imgs(name):
    file_names = os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shredded-images', name))
    file_names.sort()
    Is = []
    for f in file_names:
        I = imageio.v2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shredded-images', name, f))
        Is.append(I)
    return Is

def pixel_norm(image_right, image_left):
    height, width, channel = image_left.shape
    # Use longer bits to initialize 
    dis = np.int64(0)
    # Use the L2 norm between two schalors 
    dis = np.sum((image_right[:, 0, :] - image_left[:, width-1, :]) ** 2)
    return dis 
def pairwise_distance(Is):
    '''
    :param Is: list of N images
    :return dist: pairwise distance matrix of N x N
    
    Given a N image stripes in Is, returns a N x N matrix dist which stores the
    distance between pairs of shreds. Specifically, dist[i,j] contains the
    distance when strip j is just to the left of strip i. 
    '''
    dist = np.ones((len(Is), len(Is)))
    # Calculate the pairwise distance between two shred images 
    for i in range(len(Is)):
        for j in range(len(Is)):
            dist[i][j] = pixel_norm(Is[i], Is[j])
    
    return dist


def solve(Is):
    dist = pairwise_distance(Is)

    inds = np.arange(len(Is))
    # run greedy matching
    order = [0]
    for i in range(len(Is) - 1):
        d1 = np.min(dist[0, 1:])
        d2 = np.min(dist[1:, 0])
        if d1 < d2:
            ind = np.argmin(dist[0, 1:]) + 1
            order.insert(0, inds[ind])
            dist[0, :] = dist[ind, :]
            dist = np.delete(dist, ind, 0)
            dist = np.delete(dist, ind, 1)
            inds = np.delete(inds, ind, 0)
        else:
            ind = np.argmin(dist[1:, 0]) + 1
            order.append(inds[ind])
            dist[:, 0] = dist[:, ind]
            dist = np.delete(dist, ind, 0)
            dist = np.delete(dist, ind, 1)
            inds = np.delete(inds, ind, 0)

    return order

def combine(Is, order):
    Is = [Is[o] for o in order]
    I = np.concatenate(Is, 1)
    return I

def main(_):
    Is = load_imgs(FLAGS.test_name_simple)
    order = solve(Is)
    I = combine(Is, order)
    
    # Show concatenated image. 
    import matplotlib.pyplot as plt
    plt.imshow(I)
    plt.show()

if __name__ == '__main__':
    app.run(main)
