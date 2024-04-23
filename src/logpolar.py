import numpy as np
import cv2

N = 256
im = np.zeros((N, N, 3), dtype=np.uint8)
X, Y = np.mgrid[0:N, 0:N]

im[0:N//2, 0:N//2, 0] = 127
im[0:N, N//2:N, 1] = 127
im[N//2:N, 0:N, 2] = 127



cv2.imshow('Colors', im)

im2 = cv2.warpPolar(im, (512, 512), ((N-1)/2, (N-1)/2),  (N-1) / 2, cv2.WARP_POLAR_LOG)

cv2.imshow('Colors logpolar', im2)


cv2.waitKey()



cv2.destroyAllWindows()
