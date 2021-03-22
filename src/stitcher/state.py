import numpy as np
from ordered_set import OrderedSet
from camera import Camera
from constants import PARAMS_PER_CAMERA

class State:
  '''
  Bundle adjustment state, stores all camera parameters as a 1D array

  Warning: state.cameras getter is expensive because it recomputes cameras from params on each call
  '''

  def __init__(self, params=np.empty(0, dtype=np.float64)):
    '''
    Initialise the initial state
    '''
    if (len(params) > 0):
      self._params = params
    self._original_cameras = OrderedSet()
    self._idt = 1 # TODO: Set _idt properly (global reference camera)

  @property
  def params(self):
    return self._params


  @property
  def cameras(self):
    cameras = []

    for i in range(0, len(self._params), PARAMS_PER_CAMERA):
      new_camera = Camera(None)
      new_camera.focal = self._params[i]
      new_camera.ppx = self._params[i+1]
      new_camera.ppy = self._params[i+2]
      m = new_camera.rotvec_to_matrix(self._params[i+3:i+6])
      new_camera.R = m
      cameras.append(new_camera)

      # print(f'{i} of original_cameras {len(self._original_cameras)}')
      # print(f'==============================')
      # print(f'Original camera R: \n{self._original_cameras[i//6].R}')
      # print(f'Retrieved camera R: \n{new_camera.R}')

    return cameras


  def set_initial_cameras(self, cameras):
    self._original_cameras = cameras
    self._calculate_params(cameras)


  def updatedState(self, update):
    '''
    Returns a new state object with the update applied
    '''
    updatedParams = np.copy(self._params)
    for i in range(len(self._params)):
      if (i < self._idt * 6 + 3 or i >= self._idt * 6 + 6):
        updatedParams[i] -= update[i]
    
    return State(updatedParams)


  def _calculate_params(self, cameras):
    self._params = np.zeros((PARAMS_PER_CAMERA * len(cameras)), dtype=np.float64)

    for i in range(0, len(cameras) * PARAMS_PER_CAMERA, PARAMS_PER_CAMERA):
      camera = cameras[i//PARAMS_PER_CAMERA]
      self._params[i] = camera.focal
      self._params[i+1] = camera.ppx
      self._params[i+2] = camera.ppy
      self._params[i+3:i+6] = camera.angle_parameterisation()
      # if (self._params[i+3] == 0 and self._params[i+4] == 0 and self._params[i+5] == 0):
      #   self._idt = i