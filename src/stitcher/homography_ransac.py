import numpy as np
from homography_dlt import direct_linear_transform


def transfer_error(pt1, pt2, H):
  """
  Calculate the transfer error between the two 

  NOTE: There are other distance metrics (e.g. symmetric distance, could they be better?)

  Input:
    pt1 - Destination point
    pt2 - Point to transform onto destination plane
    H - homography to transform 

  Output: Sum squared distance error
  """

  pt1 = np.append(pt1, [1])
  pt2 = np.append(pt2, [1])

  pt1_projected = ((H @ pt1).T).T 

  diff = (pt2 - pt1_projected)

  sse = np.sum(diff**2)

  return sse


def homography_ransac(pts1, pts2, threshold, n_iterations):
  """
  Find a best guess for the homography to map pts2 onto 
  the plane of pts1
  
  Input: 
    pts1 - Destination plane
    pts2 - Points to transform onto destination plane#

  Output: Tuple of (Homography projecting pts2 onto pts1 plane, RANSAC inlier kps)
  """

  # Store maxInliners are points, if there is a tie in max, 
  # take the H that has maxInliners with the smallest standard deviation
  maxInliers = []
  bestH = None

  for i in range(n_iterations):
    # 4 random point indexes
    random_pts_idxs = np.random.choice(len(pts1), 4)

    # Get random sample using random indexes
    pts1_sample = pts1[random_pts_idxs]
    pts2_sample = pts2[random_pts_idxs]

    # Compute H using DLT
    H = direct_linear_transform(pts1_sample, pts2_sample)

    inliers = []

    # For each correspondence
    for i in range(len(pts1)):
      # Get distance for each correspondance
      distance = transfer_error(pts1[i], pts2[i], H)

      # Add correspondence to inliners if distance less than threshold
      if (distance < threshold):
        inliers.append([pts1[i], pts2[i]])

    # If inliers > maxInliers, set as new best H
    if (len(inliers) > len(maxInliers)):
      maxInliers = inliers
      bestH = H
      # TODO: else if inliers == maxInliers, pick best H based on smallest standard deviation
    
  return bestH, maxInliers
