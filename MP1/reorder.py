import os
import imageio
import numpy as np
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string('test_name_simple', 'simple_almastatue', 
                    'what set of shreads to load')

def load_imgs(name):
    file_names = os.listdir(os.path.join('shredded-images', name))
    file_names.sort()
    Is = []
    for f in file_names:
        I = imageio.v2.imread(os.path.join('shredded-images', name, f))
        Is.append(I)
    return Is


def pairwise_distance(Is):
    '''
    :param Is: list of N images
    :return dist: pairwise distance matrix of N x N
    
    Given a N image stripes in Is, returns a N x N matrix dist which stores the
    distance between pairs of shreds. Specifically, dist[i,j] contains the
    distance when strip j is just to the left of strip i. 
    '''
    dist = np.ones((len(Is), len(Is)))
    # write your code here
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
