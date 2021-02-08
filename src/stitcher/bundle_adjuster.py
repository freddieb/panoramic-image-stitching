import math
import random
import numpy as np
from numpy.core.numeric import cross
from ordered_set import OrderedSet
from state import State
from constants import PARAMS_PER_CAMERA, PARAMS_PER_POINT_MATCH, REGULARISATION_PARAM, MAX_ITR

class BundleAdjuster:
  '''
  Bundle Adjustment class that takes in matches (with initial estimates for 
  rotation and focal length of each camera) and minimises the reprojection
  error for all matches' keypoints.
  '''

  # w.r.t. K
  FOCAL_DERIVATIVE = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,0]
  ])

  # w.r.t. K
  PPX_DERIVATIVE = np.array([
    [0,0,1],
    [0,0,0],
    [0,0,0]
  ])

  # w.r.t. K
  PPY_DERIVATIVE = np.array([
    [0,0,0],
    [0,0,1],
    [0,0,0]
  ])

  def __init__(self):
    print(f'BundleAdjuster intitialised')
    self._matches = []
    self._match_count = []
    self._cameras = OrderedSet()


  def add(self, match):
    '''
    Add a match to the bundle adjuster
    '''
    num_pointwise_matches = sum(len(match.inliers) for match in self._matches)
    self._match_count.append(num_pointwise_matches)

    self._matches.append(match)
    self._cameras.update(match.cams())

    print(f'Added match {match}')


  def run(self):
    '''
    Run the bundle adjuster on the current matches to find optimal camera parameters
    '''
    if (len(self._matches) < 1):
      raise ValueError('At least one match must be added before bundle adjustment is run')

    print(f'Running bundle adjustment...')

    curr_state = State()
    curr_state.set_initial_cameras(self._cameras)

    residuals_curr = self._projection_errors(curr_state)
    error_val_curr = math.sqrt(np.mean(residuals_curr**2))
    
    print(f'Initial error: {error_val_curr}')

    itr_count = 0
    non_decrease_count = 0
    best_state = curr_state
    best_error = error_val_curr

    while (itr_count < MAX_ITR):
      J, JtJ = self._calculateJacobian(best_state)
      param_update = self._get_next_update(J, JtJ, residuals_curr)
      next_state = curr_state.updatedState(param_update)

      next_residuals = self._projection_errors(next_state)
      next_error_val = math.sqrt(np.mean(next_residuals**2))
      print(f'Next error: {next_error_val}')

      if (next_error_val >= best_error - 1e-3):
        non_decrease_count += 1
      else:
        non_decrease_count = 0
        best_error = next_error_val
        best_state = next_state
      
      if (non_decrease_count > 5):
        break
      

  def _cross_product_matrix(self, x, y, z):
    return np.array([
      [0, -z, y],
      [z, 0, -x],
      [-y, x, 0]
    ])

  def _dR_dvi(self, rotation_matrix, x, y, z):
    '''
    The derivative of the rotation with respect to each rotation parameter
    Returns 3 matrices (dR/dx, dR/dy, dR/dz)
    Calculated using https://arxiv.org/pdf/1312.0788.pdf
    '''
    ssq_params = x*x + y*y + z*z
    if (ssq_params < 1e-14):
      return np.array([
        self._cross_product_matrix(1,0,0),
        self._cross_product_matrix(0,1,0),
        self._cross_product_matrix(0,0,1)
      ])

    cross_product_matrix = self._cross_product_matrix(x, y, z)
    ret = [cross_product_matrix, cross_product_matrix, cross_product_matrix]

    ret[0] = ret[0] * x
    ret[1] = ret[1] * y
    ret[2] = ret[2] * z

    I_minus_R = np.identity(3) - rotation_matrix

    for i in range(3):
      x1, y1, z1 = np.cross(np.array([x,y,z]), I_minus_R[:,i])
      ret[i] += self._cross_product_matrix(x1, y1, z1)
      ret[i] = np.multiply(ret[i], 1 / ssq_params)
      ret[i] = ret[i] @ rotation_matrix
    
    return ret

  
  def _drdv(self, dhdv, homo, hz_inv, hz_sqr_inv):
    return np.array([
      -dhdv[0] * hz_inv + dhdv[2] * homo[0] * hz_sqr_inv,
      -dhdv[1] * hz_inv + dhdv[2] * homo[1] * hz_sqr_inv
    ])


  def _homogeneous_coordinate_2d(self, coordinate):
    '''
    Convert Cartesian coordinate to homogeneous coordinate
    '''
    return np.append(coordinate, [1])

  
  def _calculateJacobian(self, state):
    params = state.params
    cameras = state.cameras

    num_cams = len(cameras)
    num_pointwise_matches = sum(len(match.inliers) for match in self._matches)

    J = np.zeros((PARAMS_PER_POINT_MATCH * num_pointwise_matches, PARAMS_PER_CAMERA * num_cams), dtype=np.float64)
    JtJ = np.zeros((PARAMS_PER_CAMERA * num_cams, PARAMS_PER_CAMERA * num_cams), dtype=np.float64)

    all_dRdvi = []
    for i in range(len(cameras)):
      param_i = i * PARAMS_PER_CAMERA
      x, y, z = params[param_i+3:param_i+6]
      dRdvi = self._dR_dvi(cameras[i].R, x, y, z)
      all_dRdvi.append(dRdvi)

    for (i, match) in enumerate(self._matches):
      match_count_idx = self._match_count[i] * 2

      cam_to_idx = self._cameras.index(match.cam_to)
      cam_from_idx = self._cameras.index(match.cam_from)

      cam_to = cameras[cam_to_idx]
      cam_from = cameras[cam_from_idx]

      params_index_from = cam_from_idx * PARAMS_PER_CAMERA
      params_index_to = cam_to_idx * PARAMS_PER_CAMERA
      from_K = cam_from.K
      to_K_inv = np.linalg.pinv(cam_to.K)
      from_R = cam_from.R
      to_R_inv = np.linalg.pinv(cam_to.R)
      d_R_from_vi = all_dRdvi[cam_from_idx]
      d_R_to_vi = np.copy(all_dRdvi[cam_to_idx])
      d_R_to_vi_T = [m.T for m in d_R_to_vi]

      H_to_to_from = (from_K @ from_R) @ (to_R_inv @ to_K_inv)

      for (pair_index, pair) in enumerate(match.inliers):
        to_coordinate = pair[1]
        homo = H_to_to_from @ self._homogeneous_coordinate_2d(to_coordinate)
        hz_sqr_inv = 1 / math.sqrt(homo[2])
        hz_inv = 1 / homo[2]

        d_from = np.zeros((PARAMS_PER_CAMERA, PARAMS_PER_POINT_MATCH))
        d_to = np.zeros((PARAMS_PER_CAMERA, PARAMS_PER_POINT_MATCH))

        m = from_R @ to_R_inv @ to_K_inv
        dot_u2 = m @ self._homogeneous_coordinate_2d(to_coordinate)
        dot_u2[2] = 1

        d_from[0] = self._drdv(self.FOCAL_DERIVATIVE @ dot_u2, homo, hz_inv, hz_sqr_inv)
        d_from[1] = self._drdv(self.PPX_DERIVATIVE @ dot_u2, homo, hz_inv, hz_sqr_inv)
        d_from[2] = self._drdv(self.PPY_DERIVATIVE @ dot_u2, homo, hz_inv, hz_sqr_inv)

        dot_u2 = (to_R_inv @ to_K_inv) @ self._homogeneous_coordinate_2d(to_coordinate)
        dot_u2[2] = 1

        d_from[3] = self._drdv((from_K @ d_R_from_vi[0]), homo, hz_inv, hz_sqr_inv) @ dot_u2
        d_from[4] = self._drdv((from_K @ d_R_from_vi[1]), homo, hz_inv, hz_sqr_inv) @ dot_u2
        d_from[5] = self._drdv((from_K @ d_R_from_vi[2]), homo, hz_inv, hz_sqr_inv) @ dot_u2

        m = from_K @ from_R @ to_R_inv @ to_K_inv
        dot_u2 = (to_K_inv @ self._homogeneous_coordinate_2d(to_coordinate)) * -1
        dot_u2[2] = 1

        d_to[0] = self._drdv((m @ self.FOCAL_DERIVATIVE), homo, hz_inv, hz_sqr_inv) @ dot_u2
        d_to[1] = self._drdv((m @ self.PPX_DERIVATIVE), homo, hz_inv, hz_sqr_inv) @ dot_u2
        d_to[2] = self._drdv((m @ self.PPY_DERIVATIVE), homo, hz_inv, hz_sqr_inv) @ dot_u2

        m = from_K @ from_R
        dot_u2 = to_K_inv @ self._homogeneous_coordinate_2d(to_coordinate)
        dot_u2[2] = 1

        d_to[3] = self._drdv((m @ d_R_to_vi_T[0]), homo, hz_inv, hz_sqr_inv) @ dot_u2
        d_to[4] = self._drdv((m @ d_R_to_vi_T[1]), homo, hz_inv, hz_sqr_inv) @ dot_u2
        d_to[5] = self._drdv((m @ d_R_to_vi_T[2]), homo, hz_inv, hz_sqr_inv) @ dot_u2
        
        for param_idx in range(PARAMS_PER_CAMERA):
          # IS pair_index CORRECT HERE?
          J[match_count_idx, params_index_from + param_idx] = d_from[param_idx][0]
          J[match_count_idx, params_index_to + param_idx] = d_to[param_idx][0]
          J[match_count_idx+1, params_index_from + param_idx] = d_from[param_idx][1]
          J[match_count_idx+1, params_index_to + param_idx] = d_to[param_idx][1]

        for param_idx_i in range(PARAMS_PER_CAMERA):
          for param_idx_j in range(PARAMS_PER_CAMERA):
            i1 = params_index_from + param_idx_i
            i2 = params_index_to + param_idx_j
            val = np.dot(d_from[param_idx_i], d_to[param_idx_j])
            JtJ[i1][i2] += val
            JtJ[i2][i1] += val

        for param_idx_i in range(PARAMS_PER_CAMERA):
          for param_idx_j in range(param_idx_i, PARAMS_PER_CAMERA):
            i1 = params_index_from + param_idx_i
            i2 = params_index_from + param_idx_j
            val = d_to[param_idx_i] @ d_from[param_idx_j]
            JtJ[i1][i2] += val
            if (param_idx_i != param_idx_j):
              JtJ[i2][i1] += val
            
            i1 = params_index_to + param_idx_i
            i2 = params_index_to + param_idx_j
            val = np.dot(d_to[param_idx_i], d_to[param_idx_j])
            JtJ[i1][i2] += val
            if (param_idx_i != param_idx_j):
              JtJ[i2][i1] += val

        match_count_idx += 2
    
    return J, JtJ

  
  def _transform_2d(self, H, coordinate):
    '''
    Converts cartesian coordinate to homogeneous
    Project coordinate with H
    Convert back to cartesian
    '''
    homogeneous_coordinate = self._homogeneous_coordinate_2d(coordinate)
    p = H @ homogeneous_coordinate
    return np.array([p[0]/p[2], p[1]/p[2]])


  def _projection_errors(self, state):
    current_cameras = state.cameras

    num_pointwise_matches = sum(len(match.inliers) for match in self._matches)
    error = np.zeros((num_pointwise_matches * PARAMS_PER_POINT_MATCH))

    count = 0
    for match in self._matches:
      cam_from = current_cameras[self._cameras.index(match.cam_from)]
      cam_to = current_cameras[self._cameras.index(match.cam_to)]
      from_K = cam_from.K
      from_R = cam_from.R
      to_K_inv = np.linalg.pinv(cam_to.K)
      to_R_inv = np.linalg.pinv(cam_to.R)
      H_to_to_from = (from_K @ from_R) @ (to_R_inv @ to_K_inv)

      for pair in match.inliers:
        from_coordinate = pair[0]
        to_coordinate = pair[1]

        transformed = self._transform_2d(H_to_to_from, to_coordinate)
        error[count] = from_coordinate[0] - transformed[0]
        error[count+1] = from_coordinate[1] - transformed[1]

        count += 2
    
    # print(f'projection_error ({len(error)}):\n{error}')

    return error


  def _get_next_update(self, J, JtJ, residuals):
    # Regularisation
    l = random.normalvariate(1, 0.1)
    for i in range(len(self._cameras) * PARAMS_PER_CAMERA):
      if (i % PARAMS_PER_CAMERA >= 3):
        JtJ[i][i] += (3.14/16)**2 * l
      else:
        JtJ[i][i] += (1000 / 10)**2 * l

    # print(f'J.T shape: {J.T.shape}')
    # print(f'residuals: {residuals}')

    b = J.T @ residuals
    updates = np.linalg.solve(JtJ, b)
    # print(f'updates: {updates}')
    return updates
