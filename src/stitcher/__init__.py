# from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

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

def main():
  print("Call app code here...")

  MIN_MATCH_COUNT = 10

  im1path = path = "/home/fred/Documents/Uni/COMP3200_Part3Project/panoramic_image_stitching/data/imgs/01_suburbA.jpg"
  im2path = path = "/home/fred/Documents/Uni/COMP3200_Part3Project/panoramic_image_stitching/data/imgs/01_suburbB.jpg"
  im1 = cv.imread(im1path)
  im2 = cv.imread(im2path)

  im1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
  im2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

  sift = cv.SIFT_create()

  kp1, des1 = sift.detectAndCompute(im1, None)
  kp2, des2 = sift.detectAndCompute(im2, None)

  FLANN_INDEX_KDTREE = 0
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks = 50)

  flann = cv.FlannBasedMatcher(index_params, search_params)

  matches = flann.knnMatch(des1, des2, k=2)

  good = []
  for m,n in matches:
    if m.distance < 0.7*n.distance:
      good.append(m)


  if len(good) > MIN_MATCH_COUNT:
    dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # h,w = im1.shape
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # dst = cv.perspectiveTransform(pts,M)

    # im2 = cv.polylines(im2,[np.int32(dst)],True,255,3, cv.LINE_AA)

  else:
    print(f'Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}')
    matchesMask = None

  # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
  #                    singlePointColor = None,
  #                    matchesMask = matchesMask, # draw only inliers
  #                    flags = 2)

  # img3 = cv.drawMatches(im1,kp1,im2,kp2,good,None,**draw_params)  

  # print(f'im2 shape {im1.shape}')
  # print(f'im1 shape {im2.shape}')
  # print(f'input warp shape {(im1.shape[1] + im2.shape[1], im1.shape[0])}')

  # print(f'M: {M}')

  # # output shape needs to be changed based on size of both images v
  # result = cv.warpPerspective(im1, M, (im1.shape[1] + im2.shape[1], im2.shape[0]))
  # print(f'result shape {result.shape}')

  # cv.imshow("img before", result)

  # result[0:im2.shape[0], 0:im2.shape[1]] = im2

  print(M)
  result = warpTwoImages(im1, im2, M)

  cv.imshow("img after", result)
  cv.waitKey(0)

main()








# dst_pts = float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
# src_pts = float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
# M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# result = warpTwoImages(img1_color, img2_color, M)