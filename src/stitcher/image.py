import cv2 as cv

class Image:

  keypoints = None
  descriptors = None

  _sift = cv.SIFT_create()
  
  def __init__(self, img, filename):
    self._img = img
    self._filename = filename

  @property 
  def image(self):
    return self._img

  @property
  def filename(self):
    return self._filename

  def extract_sift_features(self):
    img_bw = cv.cvtColor(self._img, cv.COLOR_BGR2GRAY)
    self.keypoints, self.descriptors = self._sift.detectAndCompute(img_bw, None)
    print(f'SIFT features extracted for {self._filename}')
    
