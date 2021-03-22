import enum
import math
import random
import numpy as np
from numpy.core.numeric import cross
import re
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


  def matches(self):
    return self._matches


  def added_cameras(self):
    return self._cameras


  def add(self, match):
    '''
    Add a match to the bundle adjuster
    '''
    num_pointwise_matches = sum(len(match.inliers) for match in self._matches)
    self._match_count.append(num_pointwise_matches)

    self._matches.append(match)
    for cam in match.cams():
      self._cameras.add(cam)

    print(f'Added match {match}')


  def run(self):
    '''
    Run the bundle adjuster on the current matches to find optimal camera parameters
    '''
    if (len(self._matches) < 1):
      raise ValueError('At least one match must be added before bundle adjustment is run')

    print(f'Running bundle adjustment...')

    initial_state = State()
    initial_state.set_initial_cameras(self._cameras)

    intial_residuals = self._projection_errors(initial_state)
    intial_error = math.sqrt(np.mean(intial_residuals**2))
    
    print(f'Initial error: {intial_error}')

    print('Initial params')
    for param in initial_state.params:
      print(param)

    itr_count = 0
    non_decrease_count = 0
    best_state = initial_state
    best_residuals = intial_residuals
    best_error = intial_error

    while (itr_count < MAX_ITR):
      # print(f'[{itr_count}] Curr state: \n')
      # for (i, el) in enumerate(best_state.params):
      #   print(f'\t[{i}]: {el}')

      J, JtJ = self._calculate_jacobian(best_state)
      param_update = self._get_next_update(J, JtJ, best_residuals)
      next_state = best_state.updatedState(param_update)

      next_residuals = self._projection_errors(next_state)
      next_error_val = math.sqrt(np.mean(next_residuals**2))
      print(f'Next error: {next_error_val}')
      # return

      if (next_error_val >= best_error - 1e-3):
        non_decrease_count += 1
      else:
        print('Updating state to new best state')
        non_decrease_count = 0
        best_error = next_error_val

        # for i in range(len(best_state.params)):
        #   print(f'{best_state.params[i]} -> {next_state.params[i]}')

        best_state = next_state
        best_residuals = next_residuals;
      
      if (non_decrease_count > 5):
        break

    print(f'BEST ERROR {best_error}')

    # Update actual camera object params
    new_cameras = best_state.cameras
    for i in range(len(new_cameras)):
      # print(f'{self._cameras[i].R} = {new_cameras[i].R}')
      print(f'Final focal: {new_cameras[i].focal}')
      self._cameras[i].focal = new_cameras[i].focal
      self._cameras[i].ppx = new_cameras[i].ppx
      self._cameras[i].ppy = new_cameras[i].ppy
      self._cameras[i].R = new_cameras[i].R
      

  def _cross_product_matrix(self, x, y, z):
    return np.array([
      [0, -z, y],
      [z, 0, -x],
      [-y, x, 0]
    ], dtype=np.float64)

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
    ], dtype=np.float64)


  def _homogeneous_coordinate_2d(self, coordinate):
    '''
    Convert Cartesian coordinate to homogeneous coordinate
    '''
    return np.append(coordinate, [1])


  def _trans(self, transform, coordinate):
    if (len(coordinate) == 2):
      return self._trans(transform, self._homogeneous_coordinate_2d(coordinate))
    elif (len(coordinate) == 3):
      return transform @ coordinate

  
  def _calculate_jacobian(self, state):
    with open('ba_test_data.txt', 'w') as f:

      params = state.params
      cameras = state.cameras

      f.write('Params:\n')
      for (i, param) in enumerate(params):
        f.write(f'[{i}] {param}\n')

      f.write('\nCameras:\n')
      for (i, camera) in enumerate(cameras):
        f.write(f'[{i}] Focal: {cameras[i].focal}, R: {cameras[i].R}\n')


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
        # print(f'------------\n')
        # print(f'Loop itr: {i}')
        match_count_idx = self._match_count[i] * 2

        cam_to_idx = self._cameras.index(match.cam_to)
        cam_from_idx = self._cameras.index(match.cam_from)

        cam_to = cameras[cam_to_idx]
        cam_from = cameras[cam_from_idx]

        # print(f'from.R: {cam_from.R}')
        # print(f'to.R: {cam_to.R}')

        params_index_from = cam_from_idx * PARAMS_PER_CAMERA
        params_index_to = cam_to_idx * PARAMS_PER_CAMERA

        # print(f'params_index_from: {params_index_from}')
        # print(f'params_index_to: {params_index_to}')

        from_K = cam_from.K
        to_K_inv = np.linalg.pinv(cam_to.K)
        to_R_inv = cam_to.R.T
        from_R = cam_from.R
        d_R_from_vi = all_dRdvi[cam_from_idx]
        d_R_to_vi = np.copy(all_dRdvi[cam_to_idx])
        d_R_to_vi_T = [m.T for m in d_R_to_vi]

        H_to_to_from = (from_K @ from_R) @ (to_R_inv @ to_K_inv)
        # print(f'H_to_to_from: {H_to_to_from}')

        for (pair_index, pair) in enumerate(match.inliers):
          to_coordinate = pair[1]
          homo = self._trans(H_to_to_from, to_coordinate)
          hz_sqr_inv = 1 / (homo[2]**2)
          hz_inv = 1 / homo[2]

          d_from = np.zeros((PARAMS_PER_CAMERA, PARAMS_PER_POINT_MATCH))
          d_to = np.zeros((PARAMS_PER_CAMERA, PARAMS_PER_POINT_MATCH))

          m = from_R @ to_R_inv @ to_K_inv
          dot_u2 = self._trans(m, to_coordinate)#m @ self._homogeneous_coordinate_2d(to_coordinate)

          d_from[0] = self._drdv(self._trans(self.FOCAL_DERIVATIVE, dot_u2), homo, hz_inv, hz_sqr_inv)
          d_from[1] = self._drdv(self._trans(self.PPX_DERIVATIVE, dot_u2), homo, hz_inv, hz_sqr_inv)
          d_from[2] = self._drdv(self._trans(self.PPY_DERIVATIVE, dot_u2), homo, hz_inv, hz_sqr_inv)

          dot_u2 = self._trans((to_R_inv @ to_K_inv), to_coordinate)

          f.write(f'dot_u2: {dot_u2}\n')
          f.write(f'from_K: {from_K}\n')
          f.write(f'd_R_from_vi[0]: {d_R_from_vi[0]}\n')
          f.write(f'homo: {homo}\n')
          f.write(f'hz_inv: {hz_inv}\n')
          f.write(f'hz_sqr_inv: {hz_sqr_inv}\n')

          d_from[3] = self._drdv(self._trans((from_K @ d_R_from_vi[0]), dot_u2), homo, hz_inv, hz_sqr_inv)
          d_from[4] = self._drdv(self._trans((from_K @ d_R_from_vi[1]), dot_u2), homo, hz_inv, hz_sqr_inv)
          d_from[5] = self._drdv(self._trans((from_K @ d_R_from_vi[2]), dot_u2), homo, hz_inv, hz_sqr_inv)

          m = from_K @ from_R @ to_R_inv @ to_K_inv
          dot_u2 = self._trans(to_K_inv, to_coordinate) * -1

          # print(f'dot_u2: {dot_u2}')

          d_to[0] = self._drdv(self._trans((m @ self.FOCAL_DERIVATIVE), dot_u2), homo, hz_inv, hz_sqr_inv)
          d_to[1] = self._drdv(self._trans((m @ self.PPX_DERIVATIVE), dot_u2), homo, hz_inv, hz_sqr_inv)
          d_to[2] = self._drdv(self._trans((m @ self.PPY_DERIVATIVE), dot_u2), homo, hz_inv, hz_sqr_inv)

          # d_to[1], d_to[2] = d_to[2], d_to[1]

          m = from_K @ from_R
          dot_u2 = self._trans(to_K_inv, to_coordinate)

          d_to[3] = self._drdv(self._trans((m @ d_R_to_vi_T[0]), dot_u2), homo, hz_inv, hz_sqr_inv)
          d_to[4] = self._drdv(self._trans((m @ d_R_to_vi_T[1]), dot_u2), homo, hz_inv, hz_sqr_inv)
          d_to[5] = self._drdv(self._trans((m @ d_R_to_vi_T[2]), dot_u2), homo, hz_inv, hz_sqr_inv)

          # print(f'dfrom: {d_from}')
          # print(f'dto: {d_to}')

          f.write(f'dfrom: {d_from}\n')
          f.write(f'dto: {d_to}\n')

          # if (pair_index == 0):
          #     print(f'dfrom: {d_from}')
          #     print(f'dto: {d_to}')
          
          for param_idx in range(PARAMS_PER_CAMERA):
            # IS pair_index CORRECT HERE?
            J[match_count_idx, params_index_from + param_idx] = d_from[param_idx][0]
            # print(f'({match_count_idx}, {params_index_from + param_idx}) dfrom[{param_idx}].x: {d_from[param_idx][0]}')
            J[match_count_idx, params_index_to + param_idx] = d_to[param_idx][0]
            # print(f'({match_count_idx}, {params_index_to + param_idx}) dto[{param_idx}].x: {d_to[param_idx][0]}')
            J[match_count_idx+1, params_index_from + param_idx] = d_from[param_idx][1]
            # print(f'({match_count_idx+1}, {params_index_from + param_idx}) dfrom[{param_idx}].y: {d_from[param_idx][1]}')
            J[match_count_idx+1, params_index_to + param_idx] = d_to[param_idx][1]
            # print(f'({match_count_idx+1}, {params_index_to + param_idx}) dto[{param_idx}].y: {d_to[param_idx][1]}')

            f.write(f'({match_count_idx}, {params_index_from + param_idx}) dfrom[{param_idx}].x: {d_from[param_idx][0]}\n')
            f.write(f'({match_count_idx}, {params_index_to + param_idx}) dto[{param_idx}].x: {d_to[param_idx][0]}\n')
            f.write(f'({match_count_idx+1}, {params_index_from + param_idx}) dfrom[{param_idx}].y: {d_from[param_idx][1]}\n')
            f.write(f'({match_count_idx+1}, {params_index_to + param_idx}) dto[{param_idx}].y: {d_to[param_idx][1]}\n')

          for param_idx_i in range(PARAMS_PER_CAMERA):
            for param_idx_j in range(PARAMS_PER_CAMERA):
              # f.write(f'[l1] index_from: {params_index_from}, index_to: {params_index_to}, i: {param_idx_i}, j: {param_idx_j}\n')
              i1 = params_index_from + param_idx_i
              i2 = params_index_to + param_idx_j
              val = d_from[param_idx_i] @ d_to[param_idx_j]
              JtJ[i1][i2] += val
              JtJ[i2][i1] += val

              f.write(f'JtJ[{i1}][{i2}] += {val}\n')
              f.write(f'JtJ[{i2}][{i1}] += {val}\n')

          for param_idx_i in range(PARAMS_PER_CAMERA):
            for param_idx_j in range(param_idx_i, PARAMS_PER_CAMERA):
              # f.write(f'[l2] index_from: {params_index_from}, index_to: {params_index_to}, i: {param_idx_i}, j: {param_idx_j}\n')
              i1 = params_index_from + param_idx_i
              i2 = params_index_from + param_idx_j
              val = d_from[param_idx_i] @ d_from[param_idx_j]
              JtJ[i1][i2] += val
              f.write(f'JtJ[{i1}][{i2}] += {val}\n')
              if (param_idx_i != param_idx_j):
                JtJ[i2][i1] += val
                f.write(f'JtJ[{i2}][{i1}] += {val}\n')
              
              i1 = params_index_to + param_idx_i
              i2 = params_index_to + param_idx_j
              val = d_to[param_idx_i] @ d_to[param_idx_j]
              JtJ[i1][i2] += val
              f.write(f'JtJ[{i1}][{i2}] += {val}\n')
              if (param_idx_i != param_idx_j):
                JtJ[i2][i1] += val
                f.write(f'JtJ[{i2}][{i1}] += {val}\n')

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
      to_R_inv = cam_to.R.T
      H_to_to_from = (from_K @ from_R) @ (to_R_inv @ to_K_inv)

      start = count
      for pair in match.inliers:
        from_coordinate = pair[0]
        to_coordinate = pair[1]

        transformed = self._transform_2d(H_to_to_from, to_coordinate)
        error[count] = from_coordinate[0] - transformed[0]
        error[count+1] = from_coordinate[1] - transformed[1]

        count += 2

      print(f'Match from_{match.cam_from.image.filename} to_{match.cam_to.image.filename} error: {math.sqrt(np.mean(error[start:]**2))}')
    
    # print(f'projection_error ({len(error)}):\n{error}')

    return error


  def _get_next_update(self, J, JtJ, residuals):
    # # Regularisation
    l = random.normalvariate(1, 0.1)
    # print(f'random.normalvariate(10, 20): {random.normalvariate(10, 20)}')
    for i in range(len(self._cameras) * PARAMS_PER_CAMERA):
      if (i % PARAMS_PER_CAMERA >= 3):
        # TODO: Improve regularisation params (currently a bit off)
        JtJ[i][i] += (3.14/16) * l #random.normalvariate(10, 20) * 5000000000
      else:
        JtJ[i][i] += (1500 / 10) * l # TODO: Use intial focal estimate #random.normalvariate(10, 20) * 5000000000

    # print(f'J.T shape: {J.T.shape}')
    # print(f'residuals: {residuals}')


    # with open('test_error_residuals.txt', 'w') as f:
    #   for r in residuals:
    #     f.write(f'{r}\n')

    # openpano_JtJ = np.zeros((24,24), dtype=np.float64)
    # filename = 'ba_optimize.txt'
    # readingB = False
    # openpano_b = []
    # with open('./match_test_data/' + filename, 'r') as fp:
    #   for line in fp:
    #     if re.match(r'^\(', line):
    #       tings = [x for x in re.findall(r'\-?\d+\.?\d*e?\+?\d*', line)]
    #       # print(float(tings[2]))
    #       openpano_JtJ[int(tings[0])][int(tings[1])] = float(tings[2])
    #     elif re.match(r'b:', line):
    #       readingB = True
    #     elif (readingB and not re.match(r'^\s$', line)):
    #       bVal = [x for x in re.findall(r'\-?\d+\.?\d*e?\+?\d*', line)]
    #       # print(float(bVal[0]))
    #       openpano_b.append(float(bVal[0]))
    #     elif (readingB and re.match(r'^\s$', line)):
    #       readingB = False
    # openpano_b = np.asarray(openpano_b, dtype=np.float64)

    # with open('JtJ_test.txt', 'w') as f:
    #   print(f'JtJ.shape : {JtJ.shape}')
    #   for i in range(JtJ.shape[0]):
    #     for j in range(JtJ.shape[1]):
    #       f.write(f'({i}, {j}) {JtJ[i][j]}\n')

    b = J.T @ residuals

    # with open('JtJ_test_comparison.txt', 'w') as f:
    #   print(f'JtJ.shape : {JtJ.shape}')
    #   for i in range(JtJ.shape[0]):
    #     for j in range(JtJ.shape[1]):
    #       percentDiff = ((openpano_JtJ[i][j] - JtJ[i][j]) / openpano_JtJ[i][j]) * 100
    #       if (abs(percentDiff) > 0.001):
    #         f.write(f'({i}, {j}) JtJ={JtJ[i][j]}, OpenPano_JtJ={openpano_JtJ[i][j]} [Diff={percentDiff}]\n')

    #   f.write('\nb:\n')
    #   for (i, el) in enumerate(b):
    #     percentDiff = ((openpano_b[i] - b[i]) / openpano_b[i]) * 100
    #     if (abs(percentDiff) > 0.001):
    #       f.write(f'({i}) b={b[i]}, openpano_b={openpano_b[i]} [Diff={percentDiff}]\n')

    # JtJ = openpano_JtJ
    # b = openpano_b
    updates = np.linalg.solve(JtJ, b)
    
    # print('b:')
    # for (i, el) in enumerate(b):
    #   print(f'\t[{i}]: {el}')
    
    # print('Updates:')
    # for (i, update) in enumerate(updates):
    #   print(f'\t[{i}]: {update}')

    # print('Recomputed b vector:')
    # for (i, newB) in enumerate(JtJ@updates):
    #   print(f'\tnewB: {newB}')

    # updates = []
    # filename = 'ba_optimize.txt'
    # readingB = False
    # with open('./match_test_data/' + filename, 'r') as fp:
    #   for line in fp:
    #     if re.match(r'Update:', line):
    #       readingB = True
    #     elif (readingB and re.match(r'^\t+', line)):
    #       bVal = [x for x in re.findall(r'\-?\d+\.?\d*e?\+?\d*', line)]
    #       # print(float(bVal[0]))
    #       updates.append(float(bVal[0]))
    #     elif (readingB and re.match(r'^\s$', line)):
    #       readingB = False
    
    # print('Updates')
    # for update in updates:
    #   print(update)
    # updates = np.array(updates, dtype=np.float64)
    
    return updates
