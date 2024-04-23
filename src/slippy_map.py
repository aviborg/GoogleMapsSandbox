import numpy as np

def deg2num(lat_lon_deg: np.array, zoom=18):
  if lat_lon_deg.ndim == 1:
    lat_lon_deg = lat_lon_deg.reshape(1,2)
  lat_rad = np.deg2rad(lat_lon_deg[:, 0])
  n = 1 << int(zoom)
  xtile = (lat_lon_deg[:, 1] + 180.0) / 360.0 * n
  ytile = (1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n
  return np.transpose(np.array([xtile, ytile]))

def num2deg(x_y_tile, zoom=18):
  n = 1 << int(zoom)
  lon_deg = x_y_tile[:, 0] / n * 360.0 - 180.0
  lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * x_y_tile[:, 1] / n)))
  lat_deg = np.rad2deg(lat_rad)
  return np.transpose(np.array([lat_deg, lon_deg]))

if __name__ == '__main__':
  bb = np.array(((49.58107321890151, 15.939925928019788),
    (49.580829484421415, 15.943419078099874),
    (49.57941785825294, 15.943641732159657),
    (49.57977443302853, 15.939673044838734)))
  # bb = np.array((49.57977443302853, 15.939673044838734))
  x = deg2num(bb)
  print(x)
  b = num2deg(x)
  print(b)
  print(bb-b)
  #  print(globals()[sys.argv[1]]((float(sys.argv[2]), float(sys.argv[3]))))