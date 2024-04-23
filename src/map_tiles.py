import sys
from pathlib import Path
#from multiprocessing import Pool
import requests
import json
import time
import io
import numpy as np
import cv2
import slippy_map


class map_tiles:
  def __init__(self):
    self.imageFormat = "png"
    self.api_key_filepath = "api_key.txt"   # create an api_key at https://console.cloud.google.com/google/maps-apis/home and store it in a file before running this
    self.api_key = self.get_api_key()
    self.map_type="satellite"
    self.session_filepath = f"{self.map_type}_session_token.json"
    self.session_token = None
    self.zoom = None
    self.x = None
    self.y = None
  
  def get_xy(self):
    return self.x, self.y
  
  def get_api_key(self):
    self.api_key = None
    if Path(self.api_key_filepath).is_file():
      with open(self.api_key_filepath, "r") as api_file:
        self.api_key = api_file.read().strip()
    return self.api_key

  def get_session_token(self):
    if Path(self.session_filepath).is_file():
      with open(self.session_filepath, "r") as token_file:
        self.session_token = json.load(token_file)
      if float(self.session_token["expiry"]) < time.time() + 30 or self.session_token["imageFormat"] != self.imageFormat:
        self.session_token = None
    if self.session_token is None:
        data = {"mapType": self.map_type, "language": "en-US", "region": "US", "imageFormat": self.imageFormat}
        headers = {"Content-Type": "application/json"}
        url = f"https://tile.googleapis.com/v1/createSession?key={self.api_key}"
        session_token_response = requests.post(url, json=data, headers=headers)
        if session_token_response.ok:
          with open(self.session_filepath, "w") as token_file:
            token_file.write(session_token_response.text)
        self.session_token = json.loads(session_token_response.text)
    return self.session_token

  def get_map_tile(self, i):
    xi = i % len(self.x)
    yi = i // len(self.x)
    self.get_session_token()
    image_path = Path(str(self.map_type), str(self.zoom), str(self.x[xi]), f"{self.y[yi]}.{self.imageFormat}")
    if Path.is_file(image_path):
      im = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    else:
      url = f"https://tile.googleapis.com/v1/2dtiles/{self.zoom}/{self.x[xi]}/{self.y[yi]}?session={self.session_token["session"]}&key={self.api_key}"
      map_tile_response = requests.get(url)
      if map_tile_response.ok:
        im = cv2.imdecode(np.frombuffer(map_tile_response.content, np.uint8), cv2.IMREAD_UNCHANGED)
        Path.mkdir(image_path.parent, parents=True, exist_ok=True)
        with open(image_path, "wb") as image_file:
          image_file.write(map_tile_response.content)
    return im, i
  
  def get_map_tile_gray(self, i):
    xi = i % len(self.x)
    yi = i // len(self.x)
    self.get_session_token()
    image_path = Path(str(self.map_type), 'gray', str(self.zoom), str(self.x[xi]), f"{self.y[yi]}.{self.imageFormat}")
    if Path.is_file(image_path):
      im = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    else:
      im, i = self.get_map_tile(i)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      Path.mkdir(image_path.parent, parents=True, exist_ok=True)
      cv2.imwrite(str(image_path), im)
    return im, i

  def get_viewport(self, north=49.58311641811828, south=49.57866438815705, east=15.942535400390625, west=15.937042236328125, zoom=20):
    self.get_session_token()
    self.viewport = None
    url = f"https://tile.googleapis.com/tile/v1/viewport?session={self.session_token["session"]}&key={self.api_key}&zoom={zoom}&north={north}&south={south}&east={east}&west={west}"
    viewport_response = requests.get(url)
    if viewport_response.ok:
      self.viewport = json.loads(viewport_response.text)
    return self.viewport
  
  def set_bbox(self, bbox, zoom=18):
    if self.session_token is None:
      self.get_session_token()
    xy = slippy_map.deg2num(bbox, zoom)
    for idx in range(xy.shape[0]):
      if idx == 0:
        (xmin, ymin) = xy[idx, :]
        (xmax, ymax) = xy[idx, :]
      else:
        xmin, ymin = np.minimum((xmin, ymin), xy[idx, :])
        xmax, ymax = np.maximum((xmax, ymax), xy[idx, :])
    self.x = np.arange(int(xmin),  int(xmax) + 1)
    self.y = np.arange(int(ymin),  int(ymax) + 1)
    self.zoom = zoom

  def get_map_image(self) -> np.array:
    tile_width = self.session_token["tileWidth"]
    tile_height = self.session_token["tileHeight"]
    image = np.zeros((len(self.y) * tile_height, len(self.x) * tile_width, 3), np.uint8)
    N = len(self.x) * len(self.y)
    #with Pool(processes = 1) as pool:
    #  for im, i in pool.imap_unordered(self.get_map_tile, range(N)):
    for i in range(N):
      xi = i % len(self.x)
      yi = i // len(self.x)
      image[(yi * tile_height):((yi + 1) * tile_height), (xi * tile_width):((xi + 1) * tile_width)] = self.get_map_tile(i)[0]
    return image
  
  def get_map_image_gray(self) -> np.array:
    tile_width = self.session_token["tileWidth"]
    tile_height = self.session_token["tileHeight"]
    image = np.zeros((len(self.y) * tile_height, len(self.x) * tile_width), np.uint8)
    N = len(self.x) * len(self.y)
    #with Pool(processes = 1) as pool:
    #  for im, i in pool.imap_unordered(self.get_map_tile, range(N)):
    for i in range(N):
      xi = i % len(self.x)
      yi = i // len(self.x)
      image[(yi * tile_height):((yi + 1) * tile_height), (xi * tile_width):((xi + 1) * tile_width)] = self.get_map_tile_gray(i)[0]
    return image



if __name__ == '__main__':
  bb = np.array([[49.58107321890151, 15.939925928019788],
          [49.580829484421415, 15.943419078099874],
          [49.57941785825294, 15.943641732159657],
          [49.57977443302853, 15.939673044838734]])
  
  mtiles = map_tiles()
  mtiles.set_bbox(bb, 18)
  image = mtiles.get_map_image()
  cv2.imshow("Map tiles", image)
  cv2.waitKey()
  cv2.destroyAllWindows()
