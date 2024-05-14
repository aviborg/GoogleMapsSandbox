from pathlib import Path
import sys
import os
from map_tiles import map_tiles
import slippy_map
import time
import cv2
import numpy as np

class image_data:
  def __init__(self, image, bbox):
    self.image = np.empty([])
    self.bbox = np.empty([])

def im_8bit(image):
  return 254.0 * (image - np.min(image)) / (np.max(image) - np.min(image)) + 1

def filter_image(image, blur_size=5):
  image = np.float64(image)
  image = cv2.GaussianBlur(image, np.array([blur_size, blur_size]), 0)
  image *= cv2.createHanningWindow(image.shape[1::-1], cv2.CV_64F)  
  return im_8bit(image)

def high_pass_filter(shape, sigma=5):
  out = np.zeros(shape)
  rows = cv2.getGaussianKernel(shape[0]-1, sigma, cv2.CV_64F)
  cols = cv2.getGaussianKernel(shape[1]-1, sigma, cv2.CV_64F)
  out[1:, 1:] = np.outer(rows, cols)
  out /= np.max(out)
  return (1.0 - out)


def get_reference_bbox(bb, reference_area_ratio=2):
  centerpoint = np.mean(bb, 0)
  cross_corners = bb[0:2, :] - bb[2:4, :]
  angle_deg = np.sum(cross_corners, axis=0) 
  angle_deg = np.arctan2(angle_deg[1], angle_deg[0]) * 180.0 / np.pi
  rescaled_bb = np.zeros((4,2))
  rescaled_bb[0,:] = centerpoint + cross_corners[0,:] * reference_area_ratio / 2.0
  rescaled_bb[1,:] = centerpoint + cross_corners[1,:] * reference_area_ratio / 2.0
  rescaled_bb[2,:] = centerpoint - cross_corners[0,:] * reference_area_ratio / 2.0
  rescaled_bb[3,:] = centerpoint - cross_corners[1,:] * reference_area_ratio / 2.0
  return centerpoint, rescaled_bb, angle_deg



def world2image(latlon, image, image_xy, zoom):
  xy = slippy_map.deg2num(latlon, zoom)
  xy0 = np.array([image_xy[0][0], image_xy[1][0]])
  xy1 = np.array([image_xy[0][-1] + 1, image_xy[1][-1] + 1])
  normalized_coordinate = (xy - xy0) / (xy1 - xy0)
  return normalized_coordinate * np.tile(image.shape[1::-1], (normalized_coordinate.shape[0], 1))

def image2world(xy, image, image_xy, zoom):
  xy = slippy_map.num2deg(xy, zoom)
  xy0 = np.array([image_xy[0][0], image_xy[1][0]])
  xy1 = np.array([image_xy[0][-1] + 1, image_xy[1][-1] + 1])
  normalized_coordinate = (xy - xy0) / (xy1 - xy0)
  return np.int32(normalized_coordinate * np.tile(image.shape[1::-1], (normalized_coordinate.shape[0], 1)))

def im2logpolar(im, radius):
  im = 255 * (im - np.min(im)) / (np.max(im) - np.min(im))
  return cv2.warpPolar(im, (radius * 2, radius * 2), (np.array((im.shape[1::-1])) - 1) / 2,  (radius - 1) / 2,  cv2.INTER_AREA + cv2.WARP_POLAR_LOG)

def logpolar2anglescale(rho, phi, radius):
  angle_deg = 180 * phi / radius
  scale = np.exp(rho * np.log(radius) / (radius * 2))
  return angle_deg, scale
  


def register_image(sensor_image, bb):
  timer = time.time()
  radius = 1024
  zoom = 18
  filter_size = 5
  hpf_sigma = 20
  centerpoint, rescaled_bb, angle_deg = get_reference_bbox(bb)
  mtiles = map_tiles()
  mtiles.set_bbox(rescaled_bb, zoom)

  reference_image = filter_image(mtiles.get_map_image_gray(), filter_size)
  if False:
    reference_image = reference_image[(reference_image.shape[0] // 2 - sensor_image.shape[0] // 2):(reference_image.shape[0] // 2 + sensor_image.shape[0] // 2), (reference_image.shape[1] // 2 - sensor_image.shape[1] // 2):(reference_image.shape[1] // 2 + sensor_image.shape[1] // 2)]
  
  reference_image_fft = np.fft.fftshift(np.fft.fft2(reference_image)) 
  hpf = high_pass_filter(reference_image_fft.shape, hpf_sigma)
  im0 = np.abs(reference_image_fft) * hpf
  im0 = cv2.resize(im0, (radius, radius))
  print(time.time() - timer)
  timer = time.time()

  sensor_image = filter_image(cv2.cvtColor(sensor_image, cv2.COLOR_BGR2GRAY), filter_size)

  if False:
    sensor_image = filter_image(reference_image)
    transformMatrix = cv2.getRotationMatrix2D((np.array(sensor_image.shape[1::-1]) - 1) / 2, 10, 1/0.8)
    sensor_image = cv2.warpAffine(sensor_image, transformMatrix, (sensor_image.shape[1], sensor_image.shape[0]))
    
  target_image = np.zeros(reference_image.shape, np.float64)
  target_image[(target_image.shape[0] // 2 - sensor_image.shape[0] // 2):(target_image.shape[0] // 2 + sensor_image.shape[0] // 2), (target_image.shape[1] // 2 - sensor_image.shape[1] // 2):(target_image.shape[1] // 2 + sensor_image.shape[1] // 2)] = sensor_image
  target_image_fft = np.fft.fftshift(np.fft.fft2(target_image))
  im1 = np.abs(target_image_fft) * hpf 
  im1 = cv2.resize(im1, (radius, radius))
  
  cv2.imshow("Ref", np.uint8(im_8bit(reference_image)))
  cv2.imshow("Tar", np.uint8(im_8bit(target_image)))
    
  im0 = im2logpolar(im0, radius)
  im1 = im2logpolar(im1, radius)

  results, confidence = cv2.phaseCorrelate(im0, im1, cv2.createHanningWindow(im0.shape[::-1], cv2.CV_64F))
  angle_deg, scale = logpolar2anglescale(results[0], results[1], radius)
  print(angle_deg, scale, confidence)
  transformMatrix = cv2.getRotationMatrix2D((np.array(target_image.shape[1::-1]) - 1) / 2, angle_deg, scale)
  target_image_rotated_scaled = cv2.warpAffine(target_image, transformMatrix, (reference_image.shape[1], reference_image.shape[0]))
  cv2.imshow("TargetRS", np.uint8(target_image_rotated_scaled))
  target_image_rotated_scaled = filter_image(target_image_rotated_scaled, filter_size)
  results, confidence = cv2.phaseCorrelate(reference_image, target_image_rotated_scaled, cv2.createHanningWindow(reference_image.shape[::-1], cv2.CV_64F))
  print(results, confidence)
  transformMatrix +=  np.float64([[0,0,-results[0]],[0,0,-results[1]]])
  print(time.time() - timer)
  
  img = cv2.warpAffine(target_image, transformMatrix, (target_image.shape[1], target_image.shape[0]) )
  cv2.imshow("Final", np.uint8((img + reference_image) / 2))
  cv2.waitKey()  
  cv2.destroyAllWindows()

  return transformMatrix, confidence

if __name__ == '__main__':
  bb = np.array([[49.58107321890151, 15.939925928019788],
          [49.58072070297653, 15.943777027929109],
          [49.57941785825294, 15.943641732159657],
          [49.57977443302853, 15.939673044838734]])
  SNsize = 0.0008
  WEsize = 0.002
  bb = np.array([[SNsize, -WEsize],
                 [SNsize, WEsize],
                 [-SNsize, WEsize],
                 [-SNsize, -WEsize]])
  centerpoint = np.array([49.58024655, 15.94175443])
  bb += centerpoint
  #bb = np.array([[-2,-1], [-2,-2], [-1,-2], [-1,-1]])  
  img = cv2.imread('example/frame2rgb.png', cv2.IMREAD_UNCHANGED)



  transformMatrix, confidence = register_image(img, bb)
  print(transformMatrix, confidence)

