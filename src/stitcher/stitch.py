import numpy as np
import cv2 as cv
import math


class Stitch:

  def __init__(self, cameras):
    self._cameras = cameras
    self._stitched_img = None


  @property
  def stitched_img(self):
    return self._stitched_img


  def run(self):
    '''
    Stitch all camera images into a single image
    '''

    # Get identity image (used as ref frame)
    identity_cam = self._get_identity_cam();

    # For each non-identity image, calculate transform
    offsets = {}

    x_min_best = 9_000_000_000
    y_min_best = 9_000_000_000
    x_max_best = -9_000_000_000
    y_max_best = -9_000_000_000

    for cam in self._cameras:

      h,w = cam.image.image.shape[:2]

      pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
      H = identity_cam.K @ identity_cam.R @ cam.R.T @ np.linalg.pinv(cam.K)
      transformed_corners = cv.perspectiveTransform(pts, H)

      [x_min, y_min] = np.int32(transformed_corners.min(axis=0).ravel()) # x,y
      [x_max, y_max] = np.int32(transformed_corners.max(axis=0).ravel()) # x,y

      if (x_min < x_min_best):
        x_min_best = x_min

      if (y_min < y_min_best):
        y_min_best = y_min

      if (x_max > x_max_best):
        x_max_best = x_max
      
      if (y_max > y_max_best):
        y_max_best = y_max

      offsets[cam] = H

    # Create final sized frame to create image in
    im_x_0 = x_min_best
    im_y_0 = y_min_best
    results = []
    for (cam, H) in offsets.items():
      # if (cam == identity_cam): 
      #   continue

      im_x_shift = -im_x_0
      im_y_shift = -im_y_0

      Ht = np.array([
        [1,0,im_x_shift],
        [0,1,im_y_shift],
        [0,0,1]])

      result = cv.warpPerspective(cam.image.image,  Ht @ H, (x_max_best-x_min_best, y_max_best-y_min_best))
      results.append(result)

    final_img = results[0]
    for res in results[1:]:
      rows,cols,channels = res.shape
      res_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
      ret, mask = cv.threshold(res_gray, 0, 255, cv.THRESH_BINARY)
      mask_inv = cv.bitwise_not(mask)

      final_img_bg = cv.bitwise_and(final_img, final_img, mask=mask_inv)
      res_img_fg = cv.bitwise_and(res, res, mask=mask)

      dst = cv.add(final_img_bg, res_img_fg)
      final_img[0:rows,0:cols] = dst

    final_img[np.where((final_img==[0,0,0]).all(axis=2))] = [255,255,255]
    self._stitched_img = final_img


  def _get_identity_cam(self):
    identity_cam = None
    for cam in self._cameras:
      R = cam.R
      if ((R.shape[0] == R.shape[1]) and np.allclose(R, np.eye(R.shape[0]))):
        identity_cam = cam
        break

    if (identity_cam == None):
      raise ValueError('No identity camera found')

    return identity_cam