import numpy as np
import cv2

def im2logpolar(im, radius):
  im = 255 * (im - np.min(im)) / (np.max(im) - np.min(im))
  return cv2.warpPolar(im, (radius * 2, radius * 2), (np.array((im.shape[1::-1])) - 1) / 2,  (radius - 1) / 2,  cv2.WARP_POLAR_LOG)


N = 256
im = np.zeros((N, N//2, 3), dtype=np.uint8)
X, Y = np.mgrid[0:N, 0:N]

im[0:N//2, 0:N//4, 0] = 127
im[0:N, N//4:N//2, 1] = 127
im[N//2:N, 0:N//2, 2] = 127



cv2.imshow('Colors', im)


cv2.imshow('Colors logpolar', im2logpolar(im, 512))


cv2.waitKey()



cv2.destroyAllWindows()
