from pathlib import Path
import sys
import os
from map_tiles import map_tiles
import slippy_map
import time
import cv2
from scipy import signal
import numpy as np
import imreg_dft as ir

class image_data:
  def __init__(self, image, bbox):
    self.image = np.empty([])
    self.bbox = np.empty([])


def filter_image(image, laplace_size, blur_size):
  image = np.float64(image) * cv2.createHanningWindow(image.shape[::-1], cv2.CV_64F)
  image = cv2.GaussianBlur(image, np.array([blur_size, blur_size]), 0)
  image = 254.0 * (image - np.min(image)) / (np.max(image) - np.min(image)) + 1
  return image

def high_pass_filter(shape, sigma=1):
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

def im2logpolar(im):
  im = 255 * (im - np.min(im)) / (np.max(im) - np.min(im))
  return cv2.warpPolar(im, im.shape[::-1], np.array(im.shape[::-1]) / 2,  np.min(im.shape) / 2, cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG)

def logpolar2anglescale(rho, phi, shape):
  radius = np.min(shape) / 2
  angle_deg = 360 * phi / shape[0]
  scale = np.exp(rho * np.log(radius) / shape[1])
  return angle_deg, scale
  


def register_image(sensor_image, bb):
  timer = time.time()
  zoom = 18
  filter_size = 3
  centerpoint, rescaled_bb, angle_deg = get_reference_bbox(bb)
  mtiles = map_tiles()
  mtiles.set_bbox(rescaled_bb, zoom)

  reference_image = filter_image(mtiles.get_map_image_gray(), 0, filter_size)
  reference_image_fft = np.fft.fftshift(np.fft.fft2(reference_image)) 
  hpf = high_pass_filter(reference_image_fft.shape, 20)
  im0 = np.abs(reference_image_fft) * hpf
  
  
  sensor_image = filter_image(cv2.cvtColor(sensor_image, cv2.COLOR_BGR2GRAY), 0, filter_size)
  target_image = np.zeros(reference_image.shape, np.float64)
  target_image[(target_image.shape[0] // 2 - sensor_image.shape[0] // 2):(target_image.shape[0] // 2 + sensor_image.shape[0] // 2), (target_image.shape[1] // 2 - sensor_image.shape[1] // 2):(target_image.shape[1] // 2 + sensor_image.shape[1] // 2)] = sensor_image
  target_image_fft = np.fft.fftshift(np.fft.fft2(target_image))
  im1 = np.abs(target_image_fft) * hpf 
  

  radius = 2**9
  im0 = im2logpolar(im0)
  #cv2.imshow("Ref", np.uint8(im0))
  im1 = im2logpolar(im1)
  #cv2.imshow("Tar", np.uint8(im1))
  results, confidence = cv2.phaseCorrelate(im0, im1, cv2.createHanningWindow(im0.shape[::-1], cv2.CV_64F))
  angle_deg, scale = logpolar2anglescale(results[1], results[0], im0.shape)
  transformMatrix = cv2.getRotationMatrix2D(np.array(target_image.shape[1::-1]) / 2, -angle_deg, scale)
  target_image_rotated_scaled = cv2.warpAffine(target_image, transformMatrix, (reference_image.shape[1], reference_image.shape[0]))
  #cv2.imshow("Rotated scaled", np.uint8((target_image_rotated_scaled + reference_image) / 2))

  results, confidence = cv2.phaseCorrelate(reference_image, target_image_rotated_scaled)
  print(results, confidence)
  translationMatrix = np.float64([[1,0,-results[0]],[0,1,-results[1]]])
  target_image_final = cv2.warpAffine(target_image_rotated_scaled, translationMatrix, (reference_image.shape[1], reference_image.shape[0]))
  print(time.time() - timer)
  #cv2.imshow("Final", np.uint8((target_image_final + reference_image) / 2))

  #cv2.waitKey()
  #cv2.destroyAllWindows()
  return np.uint8((target_image_final))

'''
The bounding box must supply four corners on this fixed format:
0-------------1
|             |
|             |
|             |
3-------------2
'''
def register_image0(sensor_image, centerpoint):
  timer = time.time()
  filter_size = 5
  laplace_size = 1
  zoom = 18
  search_area = 2.0
  centerpoint = np.mean(bb, 0)
  print(centerpoint)
  cross_corners = bb[0:2, :] - bb[2:4, :]
  angle_deg = np.sum(cross_corners, axis=0)
  angle_deg = np.arctan2(angle_deg[1], angle_deg[0]) * 180.0 / np.pi
  rescaled_bb = np.zeros((4,2))
  rescaled_bb[0,:] = centerpoint + cross_corners[0,:] * search_area / 2.0
  rescaled_bb[1,:] = centerpoint + cross_corners[1,:] * search_area / 2.0
  rescaled_bb[2,:] = centerpoint - cross_corners[0,:] * search_area / 2.0
  rescaled_bb[3,:] = centerpoint - cross_corners[1,:] * search_area / 2.0
  mtiles = map_tiles()
  mtiles.set_bbox(rescaled_bb, zoom)
  reference_image = filter_image(mtiles.get_map_image_gray(), laplace_size, filter_size)

  
  sensor_image = filter_image(cv2.cvtColor(sensor_image, cv2.COLOR_BGR2GRAY), laplace_size, filter_size)
  target_image = np.zeros(reference_image.shape, np.uint8)
  target_image[(target_image.shape[0] // 2 - sensor_image.shape[0] // 2):(target_image.shape[0] // 2 + sensor_image.shape[0] // 2), (target_image.shape[1] // 2 - sensor_image.shape[1] // 2):(target_image.shape[1] // 2 + sensor_image.shape[1] // 2)] = sensor_image
  centerpoint_image = world2image(centerpoint, reference_image, mtiles.get_xy(), zoom)
  # target_image = cv2.circle(target_image, np.uint32(np.array(target_image.shape[1::-1]) // 2), 5, [255,255,0])

  #cv2.imshow("Map tiles", image)
  #cv2.waitKey()
  #cv2.destroyAllWindows()

  angle_deg = -4
  scale_factor = 1.1
  trans = np.zeros(2)
  # trans = np.array([10.91987633, -55.05088797])
  for n in range(3):
    transformMatrix = cv2.getRotationMatrix2D(np.array(target_image.shape[1::-1]) / 2, angle_deg, scale_factor)
    transformMatrix[:,2] = transformMatrix[:,2] + trans
    imgTransformedNew = cv2.warpAffine(target_image, transformMatrix, (reference_image.shape[1], reference_image.shape[0]) )
    #imgTransformedNew = cv2.circle(imgTransformedNew, np.int32(np.array(reference_image.shape[1::-1]) / 2), 5, [127,127,255])
  
    #result = cv2.phaseCorrelate(np.float32(reference_image), np.float32(imgTransformedNew))
    #print(result)
    #reference_image = cv2.resize(reference_image, (reference_image.shape[1] // 4, reference_image.shape[0] // 4), interpolation = cv2.INTER_AREA)
    #imgTransformedNew = cv2.resize(imgTransformedNew, (imgTransformedNew.shape[1] // 4, imgTransformedNew.shape[0] // 4), interpolation = cv2.INTER_AREA)
    cv2.imshow("Map tiles", np.uint8((imgTransformedNew + reference_image) / 2))
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    result = ir.similarity(np.float32(reference_image), np.float32(imgTransformedNew) )
    angle_deg += result["angle"]
    scale_factor *= result["scale"]
    trans += result["tvec"][::-1]
    '''
    result = cv2.phaseCorrelate(np.float32(reference_image), np.float32(imgTransformedNew))
    trans -= result[0]
    print(result)

  cv2.imshow("Map tiles", np.uint8((imgTransformedNew + reference_image) / 2))
  cv2.waitKey()
  cv2.destroyAllWindows()

  
    
'''
The bounding box must supply four corners on this fixed format:
0-------------1
|             |
|             |
|             |
3-------------2
'''
def register_image2(sensor_image, bb):
  timer = time.time()
  zoom = 18
  search_area = 2.0
  centerpoint = np.mean(bb, 0)
  #print(centerpoint)
  cross_corners = bb[0:2, :] - bb[2:4, :]
  angle_deg = np.sum(cross_corners, axis=0)
  angle_deg = np.arctan2(angle_deg[1], angle_deg[0]) * 180.0 / np.pi
  rescaled_bb = np.zeros((4,2))
  rescaled_bb[0,:] = centerpoint + cross_corners[0,:] * search_area / 2.0
  rescaled_bb[1,:] = centerpoint + cross_corners[1,:] * search_area / 2.0
  rescaled_bb[2,:] = centerpoint - cross_corners[0,:] * search_area / 2.0
  rescaled_bb[3,:] = centerpoint - cross_corners[1,:] * search_area / 2.0

  mtiles = map_tiles()
  mtiles.set_bbox(rescaled_bb, zoom)
  reference_image = mtiles.get_map_image()
  pts =  world2image(bb, reference_image, mtiles.get_xy(), zoom)
  pts = np.int32(pts.reshape((-1,1,2)))
  reference_image = cv2.polylines(reference_image, [pts], True , (255,0,255))
  pts =  world2image(rescaled_bb, reference_image, mtiles.get_xy(), zoom)
  pts = np.int32(pts.reshape((-1,1,2)))
  reference_image = cv2.polylines(reference_image, [pts], True , (0,255,255))

  centerpoint_image = world2image(centerpoint, reference_image, mtiles.get_xy(), zoom)
  reference_image = cv2.circle(reference_image, np.int32(centerpoint_image[0]), 5, [255,255,0])

  #cv2.imshow("Map tiles", image)
  #cv2.waitKey()
  #cv2.destroyAllWindows()

  center_translation = np.array(reference_image.shape[1::-1]) / 2 - centerpoint_image[0]
  transformMatrix = cv2.getRotationMatrix2D(centerpoint_image[0], angle_deg, 0.8)
  transformMatrix[:,2] = transformMatrix[:,2] + center_translation
  imgTransformedNew = cv2.warpAffine(reference_image, transformMatrix, (reference_image.shape[1], reference_image.shape[0]) )
  imgTransformedNew = cv2.circle(imgTransformedNew, np.int32(np.array(reference_image.shape[1::-1]) / 2), 5, [127,127,255])
  print(time.time() - timer)
  cv2.imshow("Map tiles", imgTransformedNew)
  cv2.waitKey()
  cv2.destroyAllWindows()
  

if __name__ == '__main__':
  bb = np.array([[49.58107321890151, 15.939925928019788],
          [49.58072070297653, 15.943777027929109],
          [49.57941785825294, 15.943641732159657],
          [49.57977443302853, 15.939673044838734]])

  #bb = np.array([[-2,-1], [-2,-2], [-1,-2], [-1,-1]])  
  img = cv2.imread('D:/gitprojekt/GoogleMapsSandbox/example/frame2rgb.png', cv2.IMREAD_UNCHANGED)
  centerpoint = np.array([49.58024655, 15.94175443])
  register_image(img, bb)