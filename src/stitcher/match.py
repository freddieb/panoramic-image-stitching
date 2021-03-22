import math

class Match:

  '''
  Represents a pairwise match between two images and the inlier points
  that represent a correspondence between the two images

  Notes: 
    Homography is taken as the transform of cam_from onto cam_to
    cam_from --H-> cam_to
    x(to) = H(to)(from) @ x(from)
  '''
  _homography = None 

  def __init__(self, cam_from, cam_to, h, inliers):
    self._cam_from = cam_from
    self._cam_to = cam_to
    self._homography = h
    self._inliers = inliers

  @property
  def cam_to(self):
    return self._cam_to

  @cam_to.setter
  def cam_to(self, value):
      self._cam_to = value

  @property
  def cam_from(self):
    return self._cam_from
  
  @cam_from.setter
  def cam_from(self, value):
      self._cam_from = value

  @property
  def H(self):
    return self._homography

  @H.setter
  def H(self, value):
      self._homography = value

  @property
  def inliers(self):
    return self._inliers

  def cams(self):
    return [self._cam_to, self._cam_from]

  # TODO: Why is this finding focal ~2 times bigger than actual
  # Using method from 'Construction of panoramic mosaics with global and local alignment'
  # def estimate_focal_from_homography(self):
  #   h = self._homography

  #   f0 = None
  #   f1 = None 

  #   # f0 calculations
  #   d01 = h[0][0]**2 + h[0][1]**2 - h[1][0]**2 - h[1][1]**2
  #   v01 = (h[0][1]**2 + h[0][2]**2) / d01 

  #   d02 = h[0][0]*h[1][0] + h[0][1]*h[1][1]
  #   v02 = -(h[0][2]*h[1][2]) / d02

  #   # print('d01 {}, v01 {}'.format(d01, v01))
  #   # print('d02 {}, v02 {}'.format(d02, v02))

  #   if (d01 != 0 and v01 > 0):
  #     f0 = math.sqrt(abs(v01))
  #   elif (h[0][0]*h[1][0] != h[0][1]*h[1][1] and v02 > 0):
  #     f0 = math.sqrt(abs(v02))

  #   # f1 calculations
  #   d10 = h[2][0]**2 - h[2][1]**2
  #   v10 = (h[0][1]**2 + h[1][1]**2 - h[0][0]**2 - h[0][1]**2) / d10

  #   d11 = h[2][0] * h[2][1]
  #   v11 = -(h[0][0] * h[0][1] + h[1][0] * h[1][1]) / d11

  #   # print('d10 {}, v10 {}'.format(d10, v10))
  #   # print('d11 {}, v11 {}'.format(d11, v11))
  
  #   if (d10 != 0 and v10 > 0):
  #     f1 = math.sqrt(abs(v10))
  #   elif (d11 != 0 and v11 > 0):
  #     f1 = math.sqrt(abs(v11))

  #   if (f0 and f1):
  #     return (f0 + f1) / 2
  #   elif f0:
  #     return f0
  #   elif f1:
  #     return f1
  #   else:
  #     return 0

  def estimate_focal_from_homography(self):
    h = self._homography

    f1 = None
    f0 = None

    d1 = h[2][0] * h[2][1]
    d2 = (h[2][1] - h[2][0]) * (h[2][1] + h[2][0])
    v1 = -(h[0][0] * h[0][1] + h[1][0] * h[1][1]) / d1
    v2 = (h[0][0] * h[0][0] + h[1][0] * h[1][0] - h[0][1] * h[0][1] - h[1][1] * h[1][1]) / d2
    if (v1 < v2):
      temp = v1
      v1 = v2
      v2 = temp
    if (v1 > 0 and v2 > 0):
      f1 = math.sqrt(v1 if abs(d1) > abs(d2) else v2)
    elif (v1 > 0):
      f1 = math.sqrt(v1)
    else:
      return 0

    d1 = h[0][0] * h[1][0] + h[0][1] * h[1][1]
    d2 = h[0][0] * h[0][0] + h[0][1] * h[0][1] - h[1][0] * h[1][0] - h[1][1] * h[1][1]
    v1 = -h[0][2] * h[1][2] / d1
    v2 = (h[1][2] * h[1][2] - h[0][2] * h[0][2]) / d2
    if (v1 < v2):
      temp = v1
      v1 = v2
      v2 = temp
    if (v1 > 0 and v2 > 0):
      f0 = math.sqrt(v1 if abs(d1) > abs(d2) else v2)
    elif (v1 > 0):
      f0 = math.sqrt(v1)
    else:
      return 0

    if (math.isinf(f1) or math.isinf(f0)):
      return 0

    return math.sqrt(f1 * f0)
    

