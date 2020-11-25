import numpy as np

def normalize_points(pts):
  """
  Normalize the points to remove dependence on the origin and 
  scale of the original image. This improves the solution DLT 
  provides, especially for noisy points.

  Input: 2D list of points (e.g. [[x1,y1], [x2,y2], ...])
  Output: 2D list of transformed points (e.g. [[x1',y1'], [x2',y2'], ...])
  and the homography matrix that did the transform
  """

  [x_mean, y_mean] = np.mean(pts, axis=0)
  
  # Sum of distances between points and the mean
  total_dist = np.sum(((pt[0] - x_mean)**2 + (pt[1] - y_mean)**2)**0.5 for pt in pts)
  
  avg_dist = total_dist / len(pts)

  # Scale factor
  sf = np.sqrt(2) / avg_dist

  # Transform by translation [-sf*x_mean, -sf*y_mean] and scale by [sf, sf]
  projective_transform = np.array([[sf, 0, -sf * x_mean], [0, sf, -sf * y_mean], [0, 0, 1]])

  # Pad the inner array so that it can be multiplied by the projection matrix
  ones = np.ones((1, pts.shape[0]))
  padded_pts = np.concatenate((pts.T, ones))
  
  # Apply the transform to the points 
  transformed = projective_transform @ padded_pts

  # Get x and y coordinates, and transpose to be in original form
  pts_2d = transformed[0:2].T
    
  return pts_2d, projective_transform


def compute_matrix_A(pts1, pts2):
  """
  Compute the 2n x 9 matrix A to setup the linear equations
  that allow the homography values to be found

  Input: Corresponding (matched) normalized points
  Outputs: 2n x 9 matrix A
  """

  A = []

  for i in range (0, len(pts1)):
    x, y = pts1[i]
    x_prime, y_prime = pts2[i]

    A.append([0, 0, 0, -x, -y, -1, y_prime * x, y_prime * y, y_prime])
    A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y, -x_prime])
  
  return np.asarray(A)
  

def direct_linear_transform(pts1, pts2):
  """
  Compute the homography matrix that transforms pts1 onto the plane of pts2
  via the direct linear transform (DLT) method. This includes a
  normalisation step, as described in Section 4.4 of:
  
  R. Hartley and A. Zisserman. Multiple View Geomerty in Computer 
  Vision. Cambridge University Press, second edition, 2003

  Parameters: Corresponding points for two images (pts1 and pts2)
  Output: H, the homography matrix
  """

  pts1_normalized, pts1_T = normalize_points(pts1)
  pts2_normalized, pts2_T = normalize_points(pts2)

  A = compute_matrix_A(pts1_normalized, pts2_normalized)

  # Compute the singluar value decomposition of A
  _, _, V = np.linalg.svd(A)
  
  # Get the last col of V and normalise by last value
  h = V[-1, :] / V[-1, -1]

  # Reshape to 3x3 homography matrix
  H_normalized = h.reshape(3, 3)

  # Denormalize the homography matrix
  H = np.linalg.inv(pts2_T) @ H_normalized @ pts1_T

  return H