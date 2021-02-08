import numpy as np
import cv2 as cv
import os
import argparse
import pickle
import random
import math
import copyreg

from camera_estimator import CameraEstimator
from homography_ransac import homography_ransac
from matcher import Matcher
from image import Image

######## Fixes cv.KeyPoint pickle error ##################################
def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)
##########################################################################


def skip_diag_strided(A):
  m = A.shape[0]
  strided = np.lib.stride_tricks.as_strided
  s0,s1 = A.strides
  return strided(A.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)


# https://stackoverflow.com/a/20355545
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

  result = cv.warpPerspective(img2, Ht.dot(H), (x_max-x_min, y_max-y_min))
  result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1

  return result


# Finds all SIFT keypoint matches 
def find_sift_matches(img1, img2):
  img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
  img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

  sift = cv.SIFT_create()

  kp1, des1 = sift.detectAndCompute(img1, None)
  kp2, des2 = sift.detectAndCompute(img2, None)

  # Compute putative correspondences
  FLANN_INDEX_KDTREE = 0
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks = 50)

  flann = cv.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(des1, des2, k=2)

  good = []
  for m,n in matches:
    if m.distance < 0.7 * n.distance:
      good.append(m)
  
  return good, kp1, kp2


def load_images_from_dir(dir):
  imgs = []
  for filename in sorted(os.listdir(dir)):
    img = cv.imread(os.path.join(dir, filename))
    if img is not None:
      imgs.append(img)
  return imgs

def load_images_from_dir_2(dir):
  imgs = []
  for filename in sorted(os.listdir(dir)):
    img = cv.imread(os.path.join(dir, filename))
    if img is not None:
      imgs.append(Image(img, filename))
  return imgs

def find_matches(imgs):
  # Shuffle images
  # random.shuffle(imgs)
  
  try:
    confirmed_matches = pickle.load(open('confirmed_matches.p', 'rb'))
    print('Loaded previous confirmed_matches')
  except (OSError, IOError):
    # Initialise SIFT
    sift = cv.SIFT_create()

    # Initialise approx KD tree
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Find good matches
    all_keypoints = []
    all_descriptors = []

    for img in imgs:
      keypoints, descriptors = sift.detectAndCompute(img, None)
      all_keypoints.append(keypoints)
      all_descriptors.append(descriptors)

    # Find matches for the descriptors of one image
    paired = []
    potential_pairs_matches = []
    for i in range(0, len(imgs)):
      flann.clear()

      train_descriptors = [x for j,x in enumerate(all_descriptors) if j != i]
      query_descriptors = all_descriptors[i]

      # print(f'kps len {len(all_keypoints[i])}, descs len {len(query_descriptors)}')

      flann.add(train_descriptors)
      flann.train() # might be included in the knnMatch method, so may be able to remove...
      matches = flann.knnMatch(query_descriptors, k=4)

      # print(f'len(matches): {len(matches)}')
      # print(f'len(query_descriptors): {len(query_descriptors)}')
      
      potential_pairs = np.empty((len(imgs), len(query_descriptors)), dtype=int)
      potential_pairs.fill(-1)

      for (j, nearest_neighbours) in enumerate(matches):
        # potential_pairs[[n.imgIdx if n.imgIdx < i else n.imgIdx + 1 for n in nearest_neighbours]] += 1
        # Reverse so that closest overrides further points
        for n in reversed(nearest_neighbours):
          query_img_index = n.imgIdx if n.imgIdx < i else n.imgIdx + 1
          potential_pairs[query_img_index][j] = n.trainIdx
          

      # Take 6 best matching pairs' indexes
      potential_pairs_positive_count = np.sum(np.array(potential_pairs) >= 0, axis=1) #np.count_nonzero(~np.isnan(potential_pairs), axis=1)
      # print(f'potential_pairs_nonzero_count: {potential_pairs_nonzero_count}')
      pairs = np.argsort(potential_pairs_positive_count)[::-1][:6]
      # print(f'pairs: {pairs}')
      paired.append(pairs.tolist()) 
      potential_pairs_matches.append(potential_pairs)

    print(f'Paired: {paired}')
    print('Found potential pairs, will now begin RANSAC')

    confirmed_matches = []

    for (query_img_index, img_pair_indexes) in enumerate(paired):
      for pair_index in img_pair_indexes:

        if ((pair_index, query_img_index) in confirmed_matches):
          continue

        # print(f'Comparing {query_img_index} and {pair_index}')
        if query_img_index == pair_index:
          continue

        # print()

        # print(potential_pairs_matches[query_img_index][pair_index])

        # print(f'np.array(all_keypoints[query_img_index]) shape: {np.shape(np.array(all_keypoints[query_img_index]))}')
        # print(f'potential_pairs_matches[query_img_index][pair_index] shape: {np.shape(potential_pairs_matches[query_img_index][pair_index])}')
        # print(f'np.where(potential_pairs_matches[query_img_index][pair_index] != -1) shape: {np.shape(np.where(potential_pairs_matches[query_img_index][pair_index] != -1)[0])}')

        # try:
          
        # except:
        #   print(f'np.where(potential_pairs_matches[query_img_index][pair_index] != -1)[0]: {np.where(potential_pairs_matches[query_img_index][pair_index] != -1)[0]}')
        #   raise Exception('error', 'with query_keypoints where take thing')

        query_keypoints = np.take(np.array(all_keypoints[query_img_index]), np.where(potential_pairs_matches[query_img_index][pair_index] != -1)[0]).tolist()
        target_keypoints = np.take(np.array(all_keypoints[pair_index]), potential_pairs_matches[query_img_index][pair_index][potential_pairs_matches[query_img_index][pair_index] != -1]).tolist()

        if (np.shape(query_keypoints)[0] <= 5):
          continue

        # print(f'query_keypoints: {query_keypoints}')

        # print(f'query_keypoints shape: {np.shape(query_keypoints)}')
        # print(f'target_keypoints shape: {np.shape(target_keypoints)}')

        kps1 = np.float32([ keypoint.pt for keypoint in query_keypoints ]).reshape(-1,2)
        kps2 = np.float32([ keypoint.pt for keypoint in target_keypoints ]).reshape(-1,2)

        # print(f'kps1 shape: {np.shape(kps1)}')
        # print(f'kps2 shape: {np.shape(kps2)}')

        if (np.shape(kps1)[0] < 5):
          continue

        H, inliers = homography_ransac(kps1, kps2, 4, 400)

        # The metric from the paper 'Automatic Image Stitching Using Invariant Features' does not work..
        # print(f'{query_img_index} == {pair_index} || score: {inliers / (8 + 0.3 * len(query_keypoints))} || inliers: {inliers} kps: {len(query_keypoints)}')
        # if (inliers > 8 + 0.3 * len(query_keypoints)):
        #   print(f'{Color.BOLD}Match {query_img_index} {pair_index}{Color.END}')


        if (len(inliers) > 20 and len(inliers) > 0.018 * len(query_keypoints)):
          confirmed_matches.append((query_img_index, pair_index, H, inliers))

          print(f'{Color.BOLD}Match {query_img_index} {pair_index}{Color.END}')


  # Compute focal point for all confirmed matches
  focals = []
  for (query_img_idx, pair_img_idx, H, inliers) in confirmed_matches:
    estimated_focal = estimate_focal_from_homography(H)
    print('Focal length between {} and {} is {}'.format(query_img_idx, pair_img_idx, estimated_focal))
    focals.append(estimated_focal)
  
  median_focal = np.median(focals)
  print('Median focal estimate: {}'.format(median_focal))

  
  # Build a max spanning tree from connected components (weight by number of inliers)
  confirmed_matches.sort(key=lambda el: len(el[3]))

  pickle.dump(confirmed_matches, open('confirmed_matches.p', 'wb'))
  
  
  # print(f'Confirmed matches: {sorted_matches}')

  checked = dict() # Python dict guarantee order for version >=3.7
  for match in confirmed_matches:
    camera_1 = match[0]
    camera_2 = match[1]

    identifier = None
    if (camera_1 > camera_2):
      identifier = f'{camera_1}:{camera_2}'
    else:
      identifier = f'{camera_2}:{camera_1}'
    
    if (identifier in checked):
      continue
    checked[identifier] = None


  for el in checked:
    print(f'{el}')

  # Compute rotation matrix for all confirmed matches 

  # Assume first checked element is the starting point
  # TODO: Make this logic more robust (might only be well connected to one component)
  first_checked = next(iter(checked))
  cam_0 = int(first_checked.split(':')[0])
  cam_1 = int(first_checked.split(':')[1])
  print(f'cam_0: {cam_0}, cam_1: {cam_1}')

  order = [first_checked]
  connected = [cam_0, cam_1]
  for key in checked:
    if (key in order): continue # already added the first element

    first = int(key.split(':')[0])
    second = int(key.split(':')[1])

    if (first in connected or second in connected):
      connected.append(first)
      connected.append(second)
      order.append(key)

  # Test H direction using `warp_two_images`
  # print(f'H: {confirmed_matches[0][3]}')
  # result = warp_two_images(imgs[cam_0], imgs[cam_1], confirmed_matches[0][2])

  # cv.imshow("Result", result)
  # cv.waitKey(0)

  print(order)

  camera_rotation_from_origin = dict()
  R_identity = np.array([[1,0,0],[0,1,0],[0,0,1]])
  K_initial = np.array([[median_focal,0,0], [0,median_focal,0], [0,0,1]])

  camera_rotation_from_origin[cam_0] = R_identity

  for match_key in order:
    first = int(match_key.split(':')[0])
    second = int(match_key.split(':')[1])

    if (first in camera_rotation_from_origin):
      R_prev = [camera_rotation_from_origin[x] for x in camera_rotation_from_origin if x == first][0]
      print(f'K_initial: {K_initial}')
      print(f'H: {H}')
      print(f'R_prev:\n{R_prev}')
      R_next = (np.linalg.inv(K_initial) @ H @ K_initial) @ R_prev

      # Show results, this case is first ---H--> second, so first --R--> second
      print(f'R_next:\n{R_next}')
      cv.imshow('First', imgs[first])
      cv.imshow('Second', imgs[second])
      cv.waitKey(0)


    elif (second in camera_rotation_from_origin):
      R_prev = [x for x in camera_rotation_from_origin if x == first][1]
      R_next = (np.linalg.inv(K_initial) @ np.linalg.inv(H) @ K_initial) @ R_prev

  print('Done.')


# Using method from 'Construction of panoramic mosaics with global and local alignment'
def estimate_focal_from_homography(h):
  f0 = None
  f1 = None 

  # f0 calculations
  d01 = h[0][0]**2 + h[0][1]**2 - h[1][0]**2 - h[1][1]**2
  v01 = (h[0][1]**2 + h[0][2]**2) / d01 

  d02 = h[0][0]*h[1][0] + h[0][1]*h[1][1]
  v02 = -(h[0][2]*h[1][2]) / d02

  print('d01 {}, v01 {}'.format(d01, v01))
  print('d02 {}, v02 {}'.format(d02, v02))

  if (d01 != 0 and v01 > 0):
    f0 = math.sqrt(abs(v01))
  elif (h[0][0]*h[1][0] != h[0][1]*h[1][1] and v02 > 0):
    f0 = math.sqrt(abs(v02))

  # f1 calculations
  d10 = h[2][0]**2 - h[2][1]**2
  v10 = (h[0][1]**2 + h[1][1]**2 - h[0][0]**2 - h[0][1]**2) / d10

  d11 = h[2][0] * h[2][1]
  v11 = -(h[0][0] * h[0][1] + h[1][0] * h[1][1]) / d11

  print('d10 {}, v10 {}'.format(d10, v10))
  print('d11 {}, v11 {}'.format(d11, v11))
 
  if (d10 != 0 and v10 > 0):
    f1 = math.sqrt(abs(v10))
  elif (d11 != 0 and v11 > 0):
    f1 = math.sqrt(abs(v11))

  if (f0 and f1):
    return (f0 + f1) / 2
  elif f0:
    return f0
  elif f1:
    return f1
  else:
    return 0
 


def simple_stitch(imgs):
  MIN_MATCH_COUNT = 10

  if (len(imgs) < 0):
    print('No images in given directory. Exiting.')
    return None

  result = imgs.pop()

  while (len(imgs) > 0):
    currImg = imgs.pop()

    good, kp1, kp2 = find_sift_matches(result, currImg)

    if len(good) > MIN_MATCH_COUNT:
      dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
      src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)

      H, inliers = homography_ransac(dst_pts, src_pts, 4, 500)
      print(f'inliers: {inliers}')

    else:
      print(f'Not enough good matches have been found - {len(good)}/{MIN_MATCH_COUNT}')

    result = warp_two_images(currImg, result, H)

  return result


def simple_stitch_opencv(imgs):
  MIN_MATCH_COUNT = 10

  if (len(imgs) < 0):
    print('No images in given directory. Exiting.')
    return None

  result = imgs.pop()

  while (len(imgs) > 0):
    currImg = imgs.pop()

    good, kp1, kp2 = find_sift_matches(result, currImg)

    if len(good) > MIN_MATCH_COUNT:
      dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
      src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

      M, _ = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

    else:
      print(f'Not enough good matches have been found - {len(good)}/{MIN_MATCH_COUNT}')

    result = warp_two_images(currImg, result, M)

  return result


# if __name__ == '__main__':
#   OUT_TARGET_WIDTH = 1200

#   parser = argparse.ArgumentParser()
#   parser.add_argument("dir", help="input directory for individual images")
#   args = parser.parse_args()

#   imgs = load_images_from_dir(args.dir)

#   find_matches(imgs)

  # result = simple_stitch(imgs)

  # # Resize output to make result easier to view
  # target_height = int(round(OUT_TARGET_WIDTH * result.shape[0] / result.shape[1]))
  # resized_result = cv.resize(result, (OUT_TARGET_WIDTH, target_height), interpolation=cv.INTER_LANCZOS4)

  # cv.imshow("Result", resized_result)
  # cv.waitKey(0)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("dir", help="input directory for individual images")
  args = parser.parse_args()

  imgs = load_images_from_dir_2(args.dir)

  matcher = Matcher(imgs)
  matcher.pairwise_match()

  # Visually verify matches have found appropriate homographies
  # for match in matcher.matches:
  #   result = warp_two_images(match.cam_to.image.image, match.cam_from.image.image, match.H)
  #   cv.imshow('Result', result)
  #   cv.waitKey(0)

  camera_estimator = CameraEstimator(matcher.matches)
  camera_estimator.estimate()