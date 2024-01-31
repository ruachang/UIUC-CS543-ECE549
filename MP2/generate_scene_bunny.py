import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

def get_bunny(specular=False,diffuse=0.6):

    normals = np.load('stanford_bunny/normal_bunny_tfm.npy')
    depths = np.load('stanford_bunny/depth_bunny.npy')

    Z = depths
    N = normals

    bunny_mask = Z != np.inf # 1 if bunny, 0 if background
    A = np.ones_like(Z) * 0.4
    A[bunny_mask] = diffuse
    S = np.zeros_like(Z)
    
    N[Z == np.inf] = np.array([0,0,-1]) # wall behind bunny
    Z[Z == np.inf] = 4

    if specular:
      S[bunny_mask] = 1.0
    else:
      S[bunny_mask] = 0.0

    return Z, N, A, S
  
if __name__ == '__main__':
  Z, N, A, S = get_bunny(True)
  
  fig = plt.figure(constrained_layout=False, figsize=(20, 5))
  gs1 = fig.add_gridspec(nrows=1, ncols=4, left=0.05, right=0.95, wspace=0.05, top=0.95, bottom=0.05)
  axes = []
  for i in range(4):
    axes.append(fig.add_subplot(gs1[0,i]))
    axes[-1].set_axis_off()
  
  axes[0].imshow(Z, cmap='gray', vmin=0, vmax=5)
  axes[0].set_title('Depth, Z')
  # visualizing surface normals using a commonly used color map.
  N[:,:,0] = -N[:,:,0]
  axes[1].imshow((-N[:,:,[0,1,2]]+1)/2.)
  axes[1].set_title('Surface normal, N')
  axes[2].imshow(A, cmap='gray', vmin=0, vmax=1)
  axes[2].set_title('Ambient reflection coefficient, $k_a =$ Diffuse reflection coefficient, $k_d$')
  axes[3].imshow(S, cmap='gray', vmin=0, vmax=1)
  axes[3].set_title('Specular reflection coefficient, $k_s$')
  plt.savefig('input_bunny.png', bbox_inches='tight')

