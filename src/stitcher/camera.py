import numpy as np
import math
from scipy.spatial.transform import Rotation

class Camera:

  focal = 1
  ppx = 0
  ppy = 0
  R = None

  # Constructor
  def __init__(self, image):
    self._image = image

  @property
  def image(self):
    return self._image

  @property
  def K(self):
    I = np.identity(3, dtype=np.float64)
    I[0][0] = self.focal
    I[0][2] = self.ppx
    I[1][1] = self.focal
    I[1][2] = self.ppy
    return I

  def angle_parameterisation(self):
    # rotation = Rotation.from_matrix(self.R)
    # return rotation.as_rotvec()
    u,s,v = np.linalg.svd(self.R)
    R_new = u @ (v) # TODO: might need to be transposed...
    if (np.linalg.det(R_new) < 0):
      R_new *= -1

    # print('')
    rx = R_new[2][1] - R_new[1][2]
    ry = R_new[0][2] - R_new[2][0]
    rz = R_new[1][0] - R_new[0][1]

    s = math.sqrt(rx**2 + ry**2 + rz**2)
    if (s < 1e-7):
      rx, ry, rz = 0, 0, 0
    else:
      cos = (R_new[0][0] + R_new[1][1] + R_new[2][2] - 1) * 0.5
      if (cos > 1):
        cos = 1
      elif (cos < -1):
        cos = -1
      
      theta = np.arccos(cos)
      mul = 1 / s * theta
      rx *= mul
      ry *= mul
      rz *= mul

    return np.array([rx, ry, rz], dtype=np.float64)


  # def matrix_to_rotvec(self, R):
  #   u,s,v = np.linalg.svd(R)
  #   R_new = u @ (v) # TODO: might need to be transposed...
  #   if (np.linalg.det(R_new) < 0):
  #     R_new *= -1

  #   # print('')
  #   rx = R_new[2][1] - R_new[1][2]
  #   ry = R_new[0][2] - R_new[2][0]
  #   rz = R_new[1][0] - R_new[0][1]

  #   s = math.sqrt(rx**2 + ry**2 + rz**2)
  #   if (s < 1e-7):
  #     rx, ry, rz = 0, 0, 0
  #   else:
  #     cos = (R_new[0][0] + R_new[1][1] + R_new[2][2] - 1) * 0.5
  #     if (cos > 1):
  #       cos = 1
  #     elif (cos < -1):
  #       cos = -1
      
  #     theta = np.arccos(cos)
  #     mul = 1 / s * theta
  #     rx *= mul
  #     ry *= mul
  #     rz *= mul

  #   return [rx, ry, rz]


  def rotvec_to_matrix(self, rotvec):
    rotation = Rotation.from_rotvec(rotvec)
    return rotation.as_matrix()

    # rx, ry, rz = rotvec
    # theta = rx*rx + ry*ry + rx*rz
    # if (theta < 1e-14):
    #   return np.array([
    #     [1, -rz, ry],
    #     [rz, 1, -rx],
    #     [-ry, rx, 1]
    #   ], dtype=np.float64)

    # theta = math.sqrt(theta)
    # itheta = (1/theta) if theta else 0
    # rx *= itheta
    # ry *= itheta
    # rz *= itheta

    # u_outp = [rx*rx, rx*ry, rx*rz, rx*ry, ry*ry, ry*rz, rx*rz, ry*rz, rz*rz]
    # u_crossp = [0, -rz, ry, rz, 0, -rx, -ry, rx, 0]

    # r = np.identity(3, dtype=np.float64)
    # c = np.cos(theta)
    # s = np.sin(theta)
    # c1 = 1 - c
    # r = r * c

    # for i in range(3):
    #   for j in range(3):
    #     x = i*3 + j
    #     r[i][j] += c1 * u_outp[x] + s * u_crossp[x]

    # return r



