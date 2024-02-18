import numpy as np
from generate_scene import get_ball
# from generate_scene_bunny import get_bunny
import matplotlib.pyplot as plt

# specular exponent
k_e = 50

# Change pixel (x, y) to world cooridiante (X, Y, Z)
def pixel_world(x, y, Z, cx, cy, f):
  X = (x - cx) * Z / f
  Y = (y - cy) * Z / f
  return X, Y

# Generate unit direction vector in 3D matrix(unit vector of every pixel)
def direct_mat(x1, x2):
  vec = x2 - x1
  unit_vec = vec / np.linalg.norm(vec, axis=2).reshape(vec.shape[0], vec.shape[1], 1)
  return unit_vec

def mat_cos(x1, x2):
  point_wise_2 = np.sum(x1 * x2, axis=2)
  return point_wise_2

def render(Z, N, A, S, 
           point_light_loc, point_light_strength, 
           directional_light_dirn, directional_light_strength,
           ambient_light, k_e):
  # To render the images you will need the camera parameters, you can assume
  # the following parameters. (cx, cy) denote the center of the image (point
  # where the optical axis intersects with the image, f is the focal length.
  # These parameters along with the depth image will be useful for you to
  # estimate the 3D points on the surface of the sphere for computing the
  # angles between the different directions.
  # Z: per-pixel depth N: normal A: k_a(k_d=k_a) S: k_s
  h, w = A.shape
  cx, cy = w / 2, h /2
  f = 128.

  # Flag for the type of ligh source
  if point_light_strength[0] == 0:
    POINT_LIGHT_FLAG = False
  else:
    POINT_LIGHT_FLAG = True
  # Ambient Term
  I = A * ambient_light
  real_world_coordinate = np.zeros((h, w, 3))
  for x in range(w):
    for y in range(h):
      # calculate the world coordinates of the pixel
      X, Y = pixel_world(x, y, Z[y, x], cx, cy, f)
      real_world_coordinate[y, x] = [X, Y, Z[y, x]]
  N = N / np.linalg.norm(N, axis=2).reshape(h, w, 1)
  diffuse_intense = np.zeros((h, w))
  specular_intense = np.zeros((h, w))
  if POINT_LIGHT_FLAG:
  # point light: every pixel vi is different, si is different, it equals to light pos - pixel
  # Diffuse Term
    vi = direct_mat(real_world_coordinate, point_light_loc[0])
    diffuse_intense = A * point_light_strength[0] * np.maximum(mat_cos(vi, N), 0)
  # Specular Term
    vr = direct_mat(real_world_coordinate, [0, 0, 0])
    si = -vi - 2 * (np.sum(-vi * N, axis=2)).reshape((h, w, 1)) * N
    si = si / np.linalg.norm(si, axis=2).reshape(h, w, 1)
    specular_intense = S * point_light_strength[0] * (np.maximum(mat_cos(vr, si), 0) ** k_e)
  else: 
  # directional light: every pixel vi is the same, all are the give vector. si is still different; but need to iterate 
  # through all given direnctional light sources 
    vr = direct_mat(real_world_coordinate, [0, 0, 0])
    for i in range(len(directional_light_dirn)):
  # Diffuse Term
      vi = directional_light_dirn[i] / np.linalg.norm(directional_light_dirn[i])
      diffuse_intense += A * directional_light_strength[i] * np.maximum(mat_cos(vi, N), 0)
  # Specular Term
      si = -vi - 2 * (np.sum(-vi * N, axis=2)).reshape((h, w, 1)) * N
      si = si / np.linalg.norm(si, axis=2).reshape(h, w, 1)
      specular_intense += S * directional_light_strength[i] * (np.maximum(mat_cos(vr, si), 0) ** k_e)
  I += diffuse_intense + specular_intense
      
  I = np.minimum(I, 1)*255
  I = I.astype(np.uint8)
  I = np.repeat(I[:,:,np.newaxis], 3, axis=2)
  return I

def main():
  for specular in [True, False]:
    # get_ball function returns:
    # - Z (depth image: distance to scene point from camera center, along the
    # Z-axis)
    # - N is the per pixel surface normals (N[:,:,0] component along X-axis
    # (pointing right), N[:,:,1] component along Y-axis (pointing down),
    # N[:,:,2] component along Z-axis (pointing into the scene)),
    # - A is the per pixel ambient and diffuse reflection coefficient per pixel,
    # - S is the per pixel specular reflection coefficient.
    Z, N, A, S = get_ball(specular=specular)

    # Strength of the ambient light.
    ambient_light = 0.5
    
    # For the following code, you can assume that the point sources are located
    # at point_light_loc and have a strength of point_light_strength. For the
    # directional light sources, you can assume that the light is coming _from_
    # the direction indicated by directional_light_dirn (\hat{v}_i = directional_light_dirn), and with strength
    # directional_light_strength. The coordinate frame is centered at the
    # camera, X axis points to the right, Y-axis point down, and Z-axis points
    # into the scene.
    
    # Case I: No directional light, only point light source that moves around
    # the object. 
    point_light_strength = [1.5]
    directional_light_dirn = [[1, 0, 0]]
    directional_light_strength = [0.0]
    
    fig, axes = plt.subplots(4, 4, figsize=(15,10))
    axes = axes.ravel()[::-1].tolist()
    for theta in np.linspace(0, np.pi*2, 16): 
      point_light_loc = [[10*np.cos(theta), 10*np.sin(theta), -3]]
      I = render(Z, N, A, S, point_light_loc, point_light_strength, 
                 directional_light_dirn, directional_light_strength,
                 ambient_light, k_e)
      ax = axes.pop()
      ax.imshow(I)
      ax.set_axis_off()
    plt.savefig(f'specular{specular:d}_move_point.png', bbox_inches='tight')
    plt.close()

    # Case II: No point source, just a directional light source that moves
    # around the object.
    point_light_loc = [[0, -10, 2]]
    point_light_strength = [0.0]
    directional_light_strength = [2.5]
    
    fig, axes = plt.subplots(4, 4, figsize=(15,10))
    axes = axes.ravel()[::-1].tolist()
    for theta in np.linspace(0, np.pi*2, 16): 
      directional_light_dirn = [np.array([np.cos(theta), np.sin(theta), .1])]
      directional_light_dirn[0] = \
        directional_light_dirn[0] / np.linalg.norm(directional_light_dirn[0])
      I = render(Z, N, A, S, point_light_loc, point_light_strength, 
                 directional_light_dirn, directional_light_strength,
                 ambient_light, k_e) 
      ax = axes.pop()
      ax.imshow(I)
      ax.set_axis_off()
    plt.savefig(f'specular{specular:d}_move_direction.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
  main()
