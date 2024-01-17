import os
import imageio
import numpy as np
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string('test_name_hard', 'hard_almastatue', 
                    'what set of shreads to load')


def load_imgs(name):
    file_names = os.listdir(os.path.join('shredded-images', name))
    file_names.sort()
    Is = []
    for f in file_names:
        I = imageio.v2.imread(os.path.join('shredded-images', name, f))
        Is.append(I)
    return Is


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
    return order, offsets


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
    Is = load_imgs(FLAGS.test_name_hard)
    order, offsets = solve(Is)
    I = composite(Is, order, offsets)
    import matplotlib.pyplot as plt
    plt.imshow(I)
    plt.show()

if __name__ == '__main__':
    app.run(main)
