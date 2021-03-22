import numpy as np
import argparse
import cv2 as cv
import os
import re

from camera_estimator import CameraEstimator
from camera import Camera
from match import Match
from matcher import Matcher
from image import Image


# if __name__ == '__main__':
#   JtJ = np.zeros((24,24), dtype=np.float64)

#   filename = 'ba_optimize.txt'
#   print(filename)

#   readingB = False
#   b = []

#   with open('./match_test_data/' + filename, 'r') as fp:
#     for line in fp:
#       if re.match(r'^\(', line):
#         tings = [x for x in re.findall(r'\-?\d+\.?\d*e?\+?\d*', line)]
#         # print(float(tings[2]))
#         JtJ[int(tings[0])][int(tings[1])] = float(tings[2])
#       elif re.match(r'b:', line):
#        readingB = True
#       elif (readingB and not re.match(r'^\s$', line)):
#         bVal = [x for x in re.findall(r'\-?\d+\.?\d*e?\+?\d*', line)]
#         # print(float(bVal[0]))
#         b.append(float(bVal[0]))
#       elif (readingB and re.match(r'^\s$', line)):
#         readingB = False

#     b = np.asarray(b)

#     print(f'JtJ shape: {JtJ.shape}')
#     print(f'b shape: {b.shape}')

#     updates = np.linalg.solve(JtJ, b)

#     print("Updates:")
#     print(updates)
#     # for (i, update) in enumerate(updates):
#     #   print(f'\t[{i}]: {update}')



if __name__ == '__main__':
  
  # Read in image matches
  match_cams = set()
  for filename in os.listdir('./match_test_data'):
    if re.match(r'^match_[0-9]\-[0-9].txt$', filename):
      nums = [int(x) for x in re.findall(r'\d+', filename)]
      match_cams.add(nums[0])
      match_cams.add(nums[1])

  imgs = {}
  for filename in sorted(os.listdir('../data/carmel_all')):
    img = cv.imread(os.path.join('../data/carmel_all', filename))
    if img is not None:
      imgs[filename] = img

  cams = [Camera(Image(imgs['carmel-0' + str(x) + '.png'], 'carmel-0' + str(x) + '.png')) for x in match_cams]
  
  matches = [] # (H, [[[x,y], [x',y']], ...])
  for filename in os.listdir('./match_test_data'):
    if re.match(r'^match_[0-9]\-[0-9].txt$', filename):
      match_nums = [int(x) for x in re.findall(r'\d+', filename)]
      nums = [int(x) for x in re.findall(r'\d+', filename)]
      cam_from_idx = nums[0]
      cam_to_idx = nums[1]

      # Extract match parameters
      contents = open('./match_test_data/' + filename, 'r').read()
      # print(contents.split())
      # words = re.compile("\s*").split(contents)
      h_pos_row = 0
      h_pos_col = 0
      H = np.identity(3, dtype=np.float64)
      inliers = []
      curr_inliers = []
      for (i, el) in enumerate(contents.split()):
        if (re.match(r'^\-?[0-9]+\.?[0-9]*', el)):
          value = float(el)
          if (i == 0):
            continue
          elif (i < 10):
            H[h_pos_row][h_pos_col] = value
            
            h_pos_col += 1
            if (h_pos_col % 3 == 0):
              h_pos_col = 0
              h_pos_row += 1
          elif (i == 10):
            continue
          else:
            if (len(curr_inliers) == 3):
              # to from is opposite way around in this system
              inliers.append([[curr_inliers[0], curr_inliers[1]], [curr_inliers[2], value]])
              curr_inliers = []
            else:
              curr_inliers.append(value)

      if (len(curr_inliers) > 0):
        print(filename)
        print(f'Inliers left: {curr_inliers}')
        raise Exception()
      H = np.linalg.pinv(H)
      for inlier in inliers:
        inlier[0], inlier[1] = inlier[1], inlier[0]
      matches.append(Match(cams[cam_to_idx], cams[cam_from_idx], H, inliers))

  camera_estimator = CameraEstimator(matches)
  camera_estimator.estimate()
        

  # Visually verify matches have found appropriate homographies
  # for match in matcher.matches:
  #   result = warp_two_images(match.cam_to.image.image, match.cam_from.image.image, match.H)
  #   cv.imshow('Result', result)
  #   cv.waitKey(0)

  # camera_estimator = CameraEstimator(matcher.matches)
  # camera_estimator.estimate()