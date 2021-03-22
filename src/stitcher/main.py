import numpy as np
import cv2 as cv
import os
import argparse
import copyreg

from camera_estimator import CameraEstimator
from matcher import Matcher
from image import Image
from stitch import Stitch

######## Fixes cv.KeyPoint pickle error ##################################
def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)
##########################################################################


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


def load_images_from_dir(dir):
  imgs = []
  for filename in sorted(os.listdir(dir)):
    img = cv.imread(os.path.join(dir, filename))
    if img is not None:
      imgs.append(Image(img, filename))
  return imgs


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("dir", help="input directory for individual images")
  args = parser.parse_args()

  imgs = load_images_from_dir(args.dir)

  matcher = Matcher(imgs)
  matcher.pairwise_match()

  # Visually verify matches have found appropriate homographies
  # for match in matcher.matches:
  #   result = warp_two_images(match.cam_to.image.image, match.cam_from.image.image, match.H)
  #   cv.imshow('Result', result)
  #   cv.waitKey(0)

  camera_estimator = CameraEstimator(matcher.matches)
  estimated_cameras = camera_estimator.estimate()

  stitch = Stitch(estimated_cameras)
  stitch.run()

  cv.imshow('Result', stitch.stitched_img)
  cv.imwrite('./stitched_img.png', stitch.stitched_img) 
  cv.waitKey(0)

