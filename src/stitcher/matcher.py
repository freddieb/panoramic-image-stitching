import cv2 as cv
import numpy as np
import pickle
from homography_ransac import homography_ransac
from match import Match
from camera import Camera

class Matcher:
  '''
  Matcher class for finding parwise matches between images by finding 
  corresponding SIFT keypoints between image
  '''

  def __init__(self, imgs):
    '''
    imgs = [Image]
    '''
    self._imgs = imgs
    self._matches = None
    self._cameras = [Camera(img) for img in self._imgs]


  @property
  def matches(self):
    return self._matches


  def pairwise_match(self):
    '''
    1. Extract SIFT keypoints
    2. Match keypoints
    3. Find good matches (RANSAC)
    4. Order matches by confidence 
    '''
    
    print(f'Images count: {len(self._imgs)}')
    try:
      pairwise_matches = pickle.load(open(f'pairwise_matches_{len(self._imgs)}.p', 'rb'))
      print('Loaded previous pairwise_matches')
    except (OSError, IOError):    
      for img in self._imgs:
        img.extract_sift_features()
      paired, potential_pairs_matches = self._match_keypoints()
      pairwise_matches = self._pairwise_match_images(paired, potential_pairs_matches)
      pickle.dump(pairwise_matches, open(f'pairwise_matches_{len(self._imgs)}.p', 'wb'))

    self._matches = pairwise_matches
    return pairwise_matches


  def _match_keypoints(self):
    '''
    Use an approx KD tree to find the best matches 
    '''
    # Initialise approx KD tree
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Find good matches
    all_keypoints = []
    all_descriptors = []

    for img in self._imgs:
      all_keypoints.append(img.keypoints)
      all_descriptors.append(img.descriptors)

    # Find matches for the descriptors of one image
    paired = []
    potential_pairs_matches = []
    for i in range(0, len(self._imgs)):
      flann.clear()

      train_descriptors = [x for j,x in enumerate(all_descriptors) if j != i]
      query_descriptors = all_descriptors[i]

      flann.add(train_descriptors)
      flann.train() # might be included in the knnMatch method, so may be able to remove...
      matches = flann.knnMatch(query_descriptors, k=4)

      # print(f'len(matches): {len(matches)}')
      # print(f'len(query_descriptors): {len(query_descriptors)}')
      
      potential_pairs = np.empty((len(self._imgs), len(query_descriptors)), dtype=int)
      potential_pairs.fill(-1)

      for (j, nearest_neighbours) in enumerate(matches):
        # potential_pairs[[n.imgIdx if n.imgIdx < i else n.imgIdx + 1 for n in nearest_neighbours]] += 1
        # Reverse so that closest overrides further points
        for n in reversed(nearest_neighbours):
          query_img_index = n.imgIdx if n.imgIdx < i else n.imgIdx + 1
          potential_pairs[query_img_index][j] = n.trainIdx

      # Take 6 best matching pairs' indexes
      potential_pairs_positive_count = np.sum(np.array(potential_pairs) >= 0, axis=1)
      # print(f'potential_pairs_nonzero_count: {potential_pairs_nonzero_count}')
      pairs = np.argsort(potential_pairs_positive_count)[::-1][:6]
      # print(f'pairs: {pairs}')
      paired.append(pairs.tolist()) 
      potential_pairs_matches.append(potential_pairs)

    return paired, potential_pairs_matches


  def _pairwise_match_images(self, paired, potential_pairs_matches):
    confirmed_matches = []
    all_keypoints = [img.keypoints for img in self._imgs]

    for (query_img_index, img_pair_indexes) in enumerate(paired):
      for pair_index in img_pair_indexes:

        match_names = [(match.cam_from.image.filename, match.cam_to.image.filename) for match in confirmed_matches]
        pair_filename = self._imgs[pair_index].filename
        query_img_filename = self._imgs[query_img_index].filename

        if ((query_img_filename, pair_filename) in match_names or (pair_filename, query_img_filename) in match_names):
          continue

        if query_img_index == pair_index:
          continue

        query_keypoints = np.take(np.array(all_keypoints[query_img_index]), np.where(potential_pairs_matches[query_img_index][pair_index] != -1)[0]).tolist()
        target_keypoints = np.take(np.array(all_keypoints[pair_index]), potential_pairs_matches[query_img_index][pair_index][potential_pairs_matches[query_img_index][pair_index] != -1]).tolist()

        if (np.shape(query_keypoints)[0] <= 5):
          continue

        kps1 = np.float32([ keypoint.pt for keypoint in query_keypoints ]).reshape(-1,2)
        kps2 = np.float32([ keypoint.pt for keypoint in target_keypoints ]).reshape(-1,2)

        if (np.shape(kps1)[0] < 5):
          continue

        H, inliers = homography_ransac(kps1, kps2, 4, 400) 
        # H, mask = cv.findHomography(kps1, kps2, cv.RANSAC, 5.0)
        # print(f'Homography found by OpenCV:\n{H}')
        # pretend_inliers = []
        # for i in range(len(kps1)):
        #   pretend_inliers.append([kps2[i], kps1[i]])
        # inliers = pretend_inliers

        for i in range(len(inliers)):
          inliers[i][0], inliers[i][1] = inliers[i][1], inliers[i][0]

        # The metric from the paper 'Automatic Image Stitching Using Invariant Features' does not work (probs implemented incorrectly)
        # inliers > 8 + 0.3 * len(query_keypoints)

        if (len(inliers) > 20 and len(inliers) > 0.018 * len(query_keypoints)):
          confirmed_matches.append(Match(self._cameras[pair_index], self._cameras[query_img_index], H, inliers))

          print(f'Match {query_img_index} {pair_index}')

    # Sort by number of inliers in descending order
    confirmed_matches.sort(reverse=True, key=lambda match: len(match.inliers))

    return confirmed_matches


