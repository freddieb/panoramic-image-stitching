import numpy as np
from bundle_adjuster import BundleAdjuster
from camera import Camera
from scipy.spatial.transform import Rotation

from scipy.spatial.transform import Rotation # TODO: remove when not required

class CameraEstimator:

  def __init__(self, matches):
    self._matches = matches


  def estimate(self):
    self._estimate_focal()
    self._max_span_tree_order()

    # Display order
    print(f'Match order:')
    for (i,m) in enumerate(self._matches):
      print(f'  {i} => Match {m.cam_from.image.filename} and {m.cam_to.image.filename}: {len(m.inliers)}')

    self._bundle_adjustment()


  def _estimate_focal(self):
    # Iterate through all matches and find median focal length. Set this for all cameras
    focals = []
    for match in self._matches:
      focal_estimate = match.estimate_focal_from_homography()
      if (focal_estimate != 0):
        focals.append(focal_estimate)
    median_focal = np.median(focals)

    print(f'Focal set to: {median_focal}')

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


  def _bundle_adjustment(self):
    '''
    Iteratively add each match to the bundle adjuster
    '''

    # test_rotation_m = np.array(
    #   [[ 1.03771525, -0.03411247, -0.28118836],
    #   [ 0.07859712,  1.01348254, -0.0207753 ],
    #   [ 0.70682254, -0.05688714,  0.80731615]
    # ])

    # cam = Camera(None)
    # my_rotvec = cam.matrix_to_rotvec(test_rotation_m)
    # scipy_rotvec = Rotation.from_matrix(test_rotation_m).as_rotvec()

    # print(f'my_rotvec: {my_rotvec}')
    # print(f'scipy_rotvec: {scipy_rotvec}')

    # my_matrix = cam.rotvec_to_matrix(my_rotvec)
    # scipy_matrix = Rotation.from_rotvec(scipy_rotvec).as_matrix()

    # print(f'my_matrix: \n{my_matrix}')
    # print(f'scipy_matrix: \n{scipy_matrix}')

    # test_rotation_m_2 = np.array(
    #   [[ 0.87544399, -0.05028404, -0.48069671],
    #   [ 0.05927213,  0.99823564,  0.00352427],
    #   [ 0.47967137, -0.03157722,  0.87687984]]
    # )

    # rotation = Rotation.from_matrix(test_rotation_m)
    # # rotvec = rotation.as_quat()

    # # new_rotation = Rotation.from_quat(rotvec)

    # print(f'{rotation.as_matrix() - R_new}')
    
    matches_to_add = self._matches.copy()
    added_cameras = set()
    ba = BundleAdjuster()

    # Intialise the first camera that will be used as reference frame
    first_cam = self._matches[0].cam_from
    first_cam.R = np.identity(3)
    first_cam.ppx, first_cam.ppy = 0, 0
    added_cameras.add(first_cam)

    while (len(matches_to_add) > 0):
      match = matches_to_add.pop(0)

      # Find which camera R needs to be estimated for
      if (match.cam_from in added_cameras):
        print('from -> to match found')
        cam_to_R = np.linalg.pinv(match.cam_to.K) @ match.H @ match.cam_from.K @ match.cam_from.R
        match.cam_to.R = cam_to_R
        match.cam_to.ppx, match.cam_to.ppy = 0, 0
      elif (match.cam_to in added_cameras):
        print('to -> from match found')
        # print(f'np.linalg.pinv(match.cam_from.K): {np.linalg.pinv(match.cam_from.K)}')
        # print(f'np.linalg.pinv(match.H): {np.linalg.pinv(match.H)}')
        # print(f'match.cam_from.K: {match.cam_from.K}')
        # print(f'match.cam_from.R: {match.cam_from.R}')
        cam_from_R = np.linalg.pinv(match.cam_from.K) @ np.linalg.pinv(match.H) @ match.cam_to.K @ match.cam_to.R 
        match.cam_from.R = cam_from_R
        match.cam_from.ppx, match.cam_from.ppy = 0, 0

      ba.add(match)
      added_cameras.update(match.cams())

      for (i, match) in enumerate(matches_to_add):
        # If both cameras already added, add the match to BA
        if (match.cam_from in added_cameras and match.cam_to in added_cameras):
          ba.add(matches_to_add.pop(i))
      
      ba.run()
