import numpy as np
import cv2 as cv
import os
import argparse


# https://stackoverflow.com/a/20355545
def warpTwoImages(img1, img2, H):
  '''warp img2 to img1 with homograph H'''
  h1,w1 = img1.shape[:2]
  h2,w2 = img2.shape[:2]
  pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
  pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
  pts2_ = cv.perspectiveTransform(pts2, H)
  pts = np.concatenate((pts1, pts2_), axis=0)
  [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
  [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
  t = [-xmin,-ymin]
  Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

  result = cv.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
  result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
  return result


# Finds all SIFT keypoint matches below a threshold distance
def findGoodMatches(img1, img2):
  img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
  img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

  sift = cv.SIFT_create()

  kp1, des1 = sift.detectAndCompute(img1, None)
  kp2, des2 = sift.detectAndCompute(img2, None)

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
  images = []
  for filename in sorted(os.listdir(dir)):
    img = cv.imread(os.path.join(dir, filename))
    if img is not None:
      images.append(img)
  return images

def stitch(imgs):
  MIN_MATCH_COUNT = 10

  if (len(imgs) < 0):
    print('No images in given directory. Exiting.')
    return None

  result = imgs.pop()

  while (len(imgs) > 0):
    currImg = imgs.pop()

    good, kp1, kp2 = findGoodMatches(result, currImg)

    if len(good) > MIN_MATCH_COUNT:
      dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
      src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

      M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    else:
      print(f'Not enough good matches have been found - {len(good)}/{MIN_MATCH_COUNT}')

    result = warpTwoImages(result, currImg, M)

  return result


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("dir", help="input directory for individual images")
  args = parser.parse_args()

  imgs = load_images_from_dir(args.dir)

  result = stitch(imgs)

  cv.imshow("Result", result)
  cv.waitKey(0)