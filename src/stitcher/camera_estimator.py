import numpy as np
from bundle_adjuster import BundleAdjuster
from camera import Camera
from scipy.spatial.transform import Rotation

import cv2 as cv
import pickle

from scipy.spatial.transform import Rotation # TODO: remove when not required

def warp_two_images(img1, img2, H):
  '''warp img2 to img1 with homography H'''
  h1,w1 = img1.shape[:2]
  h2,w2 = img2.shape[:2]

  pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
  pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)

  pts2_ = cv.perspectiveTransform(pts2, H)
  pts = np.concatenate((pts1, pts2_), axis=0)

  [x_min, y_min] = np.int32(pts.min(axis=0).ravel())
  [x_max, y_max] = np.int32(pts.max(axis=0).ravel())

  t = [-x_min,-y_min]
  Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate by t

  # b_channel, g_channel, r_channel = cv.split(img1)
  # alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 #creating a dummy alpha channel image.
  # img1 = cv.merge((b_channel, g_channel, r_channel, alpha_channel))

  result = cv.warpPerspective(img2, Ht.dot(H), (x_max-x_min, y_max-y_min))
  result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1

  return result


class CameraEstimator:

  def __init__(self, matches):
    self._matches = matches

    # for match in self._matches:
      # match.cam_from, match.cam_to = match.cam_to, match.cam_from
      # match.H = np.linalg.pinv(match.H)
      # self._normalise_match_H(match)
      # match.cam_from, match.cam_to = match.cam_to, match.cam_from
      # print(f'H: {match.H}\n')
      # for inlier in match.inliers:
      #   inlier[0], inlier[1] = inlier[1], inlier[0]


  def estimate(self):

    for m in self._matches:
      print(f'Match (unordered) {m.cam_from.image.filename} and {m.cam_to.image.filename}: {len(m.inliers)}')
      # for match in matcher.matches:
      # result = warp_two_images(m.cam_to.image.image, m.cam_from.image.image, m.H)
      # cv.imshow('Result', result)
      # cv.waitKey(0)

    self._estimate_focal()
    add_order = self._max_span_tree_order_2()

    # Display order
    print(f'Add order:')
    for (i,m) in enumerate(add_order):
      print(f'  {i} => Match {m.cam_from.image.filename} and {m.cam_to.image.filename}: {len(m.inliers)}')
    # return

    self._bundle_adjustment(add_order)

    return self._all_cameras()


  def _estimate_focal(self):
    # Iterate through all matches and find median focal length. Set this for all cameras
    focals = []
    for match in self._matches:
      focal_estimate = match.estimate_focal_from_homography()
      if (focal_estimate != 0):
        print(f'{match.cam_from.image.filename} to {match.cam_to.image.filename}: {focal_estimate}')
        focals.append(focal_estimate)
    median_focal = np.median(focals)

    print(f'Focal set to: {median_focal}')

    # if (len(focals) == 0):
    median_focal = 1580

    for camera in self._all_cameras():
      camera.focal = median_focal


  def _all_cameras(self):
    all_cameras = set()

    for match in self._matches:
      all_cameras.add(match.cam_from)
      all_cameras.add(match.cam_to)

    return all_cameras


  def _max_span_tree_order(self):
    '''
    Finds a maximum spanning tree from all matches and sorts
    them into most connected matches in descending order. This
    is the optimal order to add them to the BundleAdjuster
    '''
    sorted_matches = [self._matches[0]]
    connected_cameras = set(self._matches[0].cams())
    while (len(sorted_matches) < len(self._matches)):
      for match in self._matches[1:]:
        if (match.cam_to in connected_cameras or match.cam_from in connected_cameras):
          sorted_matches.append(match)
          connected_cameras.update(match.cams())

    self._matches = sorted_matches


  def _reverse_match(self, match):
    match.cam_from, match.cam_to = match.cam_to, match.cam_from
    match.H = np.linalg.pinv(match.H)
    for inlier in match.inliers:
      inlier[0], inlier[1] = inlier[1], inlier[0]
    self._normalise_match_H(match)


  def _normalise_match_H(self, match):
    match.H = np.multiply(match.H, 1 / match.H[2][2])


  def _max_span_tree_order_2(self):
    '''
    Finds a maximum spanning tree from all matches, with most connected edge as start point
    '''
    connected_nodes = set()
    all_cameras = self._all_cameras()
    sorted_all_cameras = sorted(all_cameras, key=lambda c:c.image.filename)
    [print(c.image.filename) for c in sorted_all_cameras]
    sorted_edges = sorted(self._matches, key=lambda m: len(m.inliers), reverse=True)
    [print(f'{e.cam_from.image.filename} - {e.cam_to.image.filename}: {len(e.inliers)}') for e in sorted_edges]
    best_edge = sorted_edges.pop(0)

    # if (sorted_all_cameras.index(best_edge.cam_from) > sorted_all_cameras.index(best_edge.cam_to)):
    #   print("edge swapped")
    #   self._reverse_match(best_edge)
    # else:
    #   self._normalise_match_H(best_edge)

    print(f'Best edge: {best_edge.cam_from.image.filename} - {best_edge.cam_to.image.filename}: {len(best_edge.inliers)}')
    print(f'Best edge H: {best_edge.H}')

    add_order = [best_edge]
    connected_nodes.add(best_edge.cam_from)
    connected_nodes.add(best_edge.cam_to)

    while (len(connected_nodes) < len(sorted_all_cameras)):
      for (i, match) in enumerate(sorted_edges):
        if (match.cam_from in connected_nodes):
          # Add node as is
          edge = sorted_edges.pop(i)
          self._normalise_match_H(edge)
          add_order.append(edge)
          connected_nodes.add(edge.cam_from)
          connected_nodes.add(edge.cam_to)
          break
        elif (match.cam_to in connected_nodes):
          # Reverse node and add
          edge = sorted_edges.pop(i)

          self._reverse_match(edge)

          add_order.append(edge)
          connected_nodes.add(edge.cam_from)
          connected_nodes.add(edge.cam_to)
          break
    
    return add_order



  def _bundle_adjustment(self, add_order):
    '''
    Iteratively add each match to the bundle adjuster
    '''
    
    # #-------start-----------
    # matches_to_add = self._matches.copy()
    # added_cameras = set()
    # ba = BundleAdjuster()

    # # Intialise the first camera that will be used as reference frame
    # first_cam = self._matches[0].cam_from
    # first_cam.R = np.identity(3)
    # first_cam.ppx, first_cam.ppy = 0, 0
    # added_cameras.add(first_cam)

    # while (len(matches_to_add) > 0):
    #   match = matches_to_add.pop(0)

    #   # Find which camera R needs to be estimated for
    #   if (match.cam_from in added_cameras):
    #     # cam_to_R = np.linalg.pinv(match.cam_to.K) @ match.H @ match.cam_from.K @ match.cam_from.R
    #     cam_to_R = (np.linalg.pinv(match.cam_from.R) @ (np.linalg.pinv(match.cam_from.K) @ match.H @ match.cam_to.K)).T
    #     match.cam_to.R = cam_to_R
    #     match.cam_to.ppx, match.cam_to.ppy = 0, 0
    #     print(f'{match.cam_from.image.filename} to {match.cam_to.image.filename}:\n {match.cam_to.R}\n\n')
    #   # elif (match.cam_to in added_cameras):
    #   #   print('to -> from match found')
    #   #   # print(f'np.linalg.pinv(match.cam_from.K): {np.linalg.pinv(match.cam_from.K)}')
    #   #   # print(f'np.linalg.pinv(match.H): {np.linalg.pinv(match.H)}')
    #   #   # print(f'match.cam_from.K: {match.cam_from.K}')
    #   #   # print(f'match.cam_from.R: {match.cam_from.R}')
    #   #   cam_from_R = np.linalg.pinv(match.cam_from.K) @ np.linalg.pinv(match.H) @ match.cam_to.K @ match.cam_to.R 
    #   #   match.cam_from.R = cam_from_R
    #   #   match.cam_from.ppx, match.cam_from.ppy = 0, 0
    #   #   print(f'{match.cam_from.image.filename} - {match.cam_from.R}')

    #   # ba.add(match)
    #   added_cameras.update(match.cams())

      # for (i, match) in enumerate(matches_to_add):
      #   # If both cameras already added, add the match to BA
      #   if (match.cam_from in added_cameras and match.cam_to in added_cameras):
      #     ba.add(matches_to_add.pop(i))
      
    #   #ba.run()
    #   #-----end---------

    ba = BundleAdjuster()

    other_matches = set(self._matches) - set(add_order)

    # print(f'Total matches: {len(self._matches)}')
    # print(f'add_order count: {len(add_order)}')
    # print(f'other_matches count: {len(other_matches)}')

    identity_cam = add_order[0].cam_from
    identity_cam.R = np.identity(3)
    identity_cam.ppx, identity_cam.ppy = 0, 0

    print(f'Identity cam: {identity_cam.image.filename}')

    print('Original match params:')
    for match in add_order:
      print(f'{match.cam_from.image.filename} to {match.cam_to.image.filename}:\n {match.cam_to.R}\n')
    print('------------------')

    for match in add_order:
      # result = warp_two_images(match.cam_from.image.image, match.cam_to.image.image, match.H)
      # cv.imshow('Original H from RANSAC', result)
      print(f'match.cam_from.R: {match.cam_from.R}')
      print(f'match.cam_from.K: {match.cam_from.K}')
      print(f'match.H: {match.H}')
      print(f'match.cam_to.K: {match.cam_to.K}')

      match.cam_to.R = (match.cam_from.R.T @ (np.linalg.pinv(match.cam_from.K) @ match.H @ match.cam_to.K)).T
      match.cam_to.ppx, match.cam_to.ppy = 0, 0
      print(f'{match.cam_from.image.filename} to {match.cam_to.image.filename}:\n {match.cam_to.R}\n')

      # reconstructed_H = match.cam_from.K @ match.cam_from.R @ match.cam_to.R.T @ np.linalg.pinv(match.cam_to.K)
      # result = warp_two_images(match.cam_from.image.image, match.cam_to.image.image, reconstructed_H)
      # cv.imshow('Result with reconstructed H', result)

      # # Matrix -> rotvec -> matrix
      # converted_R = match.cam_to.rotvec_to_matrix(match.cam_to.angle_parameterisation())
      # print(f'homography:\n{match.H}')
      # print(f'converted_R matrix:\n{converted_R}')
      # reconstructed_from_converted_R_H = match.cam_from.K @ match.cam_from.R @ converted_R.T @ np.linalg.pinv(match.cam_to.K)
      # result = warp_two_images(match.cam_from.image.image, match.cam_to.image.image, reconstructed_from_converted_R_H)
      # cv.imshow('Result with reconstructed H from converted R', result)
      # cv.waitKey(0)
      # return

      ba.add(match)

      added_cams = ba.added_cameras()
      to_add = set()
      for other_match in other_matches:
        # If both cameras already added, add the match to BA
        if (other_match.cam_from in added_cams and other_match.cam_to in added_cams):
          to_add.add(other_match)
      for match in to_add:
        # self._reverse_match(match)
        ba.add(match)
        other_matches.remove(match)
    # return
    
    all_cameras = None
    try:
      all_cameras = pickle.load(open(f'all_cameras_{len(self._all_cameras())}.p', 'rb'))

      for match in self._matches:
        for cam in all_cameras:
          if (match.cam_to.image.filename == cam.image.filename):
            match.cam_to = cam
          elif (match.cam_from.image.filename == cam.image.filename):
            match.cam_from = cam

    except (OSError, IOError):    
      ba.run()
      all_cameras = self._all_cameras()
      pickle.dump(all_cameras, open(f'all_cameras_{len(self._all_cameras())}.p', 'wb'))

    print('BA complete.')

    return

    # print('Showing new homographies')

    # # Get identity camera
    # identity_cam = None
    # for cam in all_cameras:
    #   R = cam.R
    #   if ((R.shape[0] == R.shape[1]) and np.allclose(R, np.eye(R.shape[0]))):
    #     identity_cam = cam
    #     break
    
    # if (identity_cam == None):
    #   raise ValueError('No identity camera found')

    # # panoImg = identity_cam.image.image
    # # Iterate through all non-identity cameras, adding to final panorama image
    # for cam in all_cameras:
    #   if (cam != identity_cam):
    #     print(f'identity_cam.K: {identity_cam.K}')
    #     print(f'cam.K: {cam.K}')
    #     constructed_from_converted_R_H = identity_cam.K @ identity_cam.R @ cam.R.T @ np.linalg.pinv(cam.K)
    #     result = warp_two_images(identity_cam.image.image, cam.image.image, constructed_from_converted_R_H)
    #     cv.imshow(f'Match {identity_cam.image.filename} to {match.cam_to.image.filename}, H from R', result)
    #     cv.waitKey(0)
    
    # for match in self._matches:
    #   constructed_from_converted_R_H = match.cam_from.K @ match.cam_from.R @ match.cam_to.R.T @ np.linalg.pinv(match.cam_to.K)
    #   result = warp_two_images(match.cam_from.image.image, match.cam_to.image.image, constructed_from_converted_R_H)
    #   cv.imshow(f'Match {match.cam_from.image.filename} to {match.cam_to.image.filename}, H from R', result)
    #   cv.waitKey(0)

      # return

    # print('---------------------------------')
    # print("------ Actual BA ----------------")
    # testBA = BundleAdjuster()
    # testMatches = ba.matches()

    # testBA.add(testMatches[3])
    # testBA.add(testMatches[4])
    # testBA.add(testMatches[1])
    # testBA.add(testMatches[2])
    # self._reverse_match(testMatches[0])
    # testBA.add(testMatches[0])

    # testBA.run()
