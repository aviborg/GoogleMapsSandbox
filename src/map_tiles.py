import sys
from pathlib import Path
import requests
import json
import time
import io
import numpy
from PIL import Image, ImageDraw
import slippy_map


class map_tiles:
  def __init__(self):
    self.imageFormat = "png"
    self.api_key_filepath = "api_key.txt"   # create an api_key at https://console.cloud.google.com/google/maps-apis/home and store it in a file before running this
    self.api_key = self.get_api_key()
    self.map_type="satellite"
    self.session_filepath = f"{self.map_type}_session_token.json"
    self.session_token = None
  
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

  def get_map_tile(self, x=6294, y=13288, z=15):
    self.get_session_token()
    image_path = Path(str(self.map_type), str(z), str(x), f"{y}.{self.imageFormat}")
    if Path.is_file(image_path):
      im = Image.open(image_path)
    else:
      url = f"https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}?session={self.session_token["session"]}&key={self.api_key}"
      map_tile_response = requests.get(url)
      if map_tile_response.ok:
        im = Image.open(io.BytesIO(map_tile_response.content))
        Path.mkdir(image_path.parent, parents=True, exist_ok=True)
        with open(image_path, "wb") as image_file:
          image_file.write(map_tile_response.content)
    return im

  def get_viewport(self, north=49.58311641811828, south=49.57866438815705, east=15.942535400390625, west=15.937042236328125, zoom=20):
    self.get_session_token()
    self.viewport = None
    url = f"https://tile.googleapis.com/tile/v1/viewport?session={self.session_token["session"]}&key={self.api_key}&zoom={zoom}&north={north}&south={south}&east={east}&west={west}"
    viewport_response = requests.get(url)
    if viewport_response.ok:
      self.viewport = json.loads(viewport_response.text)
    return self.viewport

  def get_map_image(self, bbox, zoom=18) -> Image:
    timer = time.time()
    if self.session_token is None:
      self.get_session_token()
    for idx, (lat, lon) in enumerate(bbox):
      x, y = slippy_map.deg2num((lat, lon), zoom)
      if idx == 0:
        (xmin, ymin) = (x, y)
        (xmax, ymax) = (x, y)
      else:
        xmin, ymin = numpy.minimum((xmin, ymin), (x, y))
        xmax, ymax = numpy.maximum((xmax, ymax), (x, y))
    x = range(int(xmin),  int(xmax) + 1)
    y = range(int(ymin),  int(ymax) + 1)
    tile_width = self.session_token["tileWidth"]
    tile_height = self.session_token["tileHeight"]
    image = Image.new("RGB", (len(x) * tile_width, len(y) * tile_height))
    for xi, xn in enumerate(x):
      for yi, yn in enumerate(y):
          image.paste(self.get_map_tile(xn, yn, zoom), (xi * tile_width, yi * tile_height))
    print(time.time() - timer)
    return image



if __name__ == '__main__':
  bb = ((49.58107321890151, 15.939925928019788),
    (49.580829484421415, 15.943419078099874),
    (49.57941785825294, 15.943641732159657),
    (49.57977443302853, 15.939673044838734))
  image = map_tiles().get_map_image(bb, 18)
  draw = ImageDraw.Draw(image)
  
  image.show()
