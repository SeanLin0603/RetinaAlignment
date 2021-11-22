import cv2
from align_main import Align


image = cv2.imread('0.jpg')
alignImg = Align(image)

print(image.shape)
print(alignImg.shape)
cv2.imwrite("0_align.jpg", alignImg)