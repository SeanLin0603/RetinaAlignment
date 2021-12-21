# RetinaAlignment

* (Resnet50_Final.pth):
    https://reurl.cc/aNbNZl
    
    
### Usage:
* example.py
``` python=1
import cv2
from retina_align import Align

image = cv2.imread('input.jpg')
alignImg, drawImg = Align(image)

print('[Info] Input shape: {}'.format(image.shape))
print('[Info] Output shape: {}'.format(alignImg.shape))
cv2.imwrite("output_draw.jpg", drawImg)
cv2.imwrite("output_align.jpg", alignImg)

```
    
* input.jpg
![](https://i.imgur.com/J6d5JhI.jpg)

* output_draw.jpg
![](https://i.imgur.com/N4XGmcO.jpg)

* output_aligned.jpg
![](https://i.imgur.com/QZ3HMxL.jpg)
