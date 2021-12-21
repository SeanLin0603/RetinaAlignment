import cv2
from retina_align import Align

image = cv2.imread('input.jpg')
alignImg, drawImg = Align(image)

print('[Info] Input shape: {}'.format(image.shape))
print('[Info] Output shape: {}'.format(alignImg.shape))
cv2.imwrite("output_draw.jpg", drawImg)
cv2.imwrite("output_align.jpg", alignImg)