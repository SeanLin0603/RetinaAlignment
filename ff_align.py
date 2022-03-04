from genericpath import exists
import os
import cv2
from pathlib import Path
from imutils import paths
from retina_align import Align

# srcDir = '/home/sean/Documents/Forensics/RealFF/c23/original_images/'
# dstDir = '/home/sean/Documents/Forensics/RealFF/c23/images/'
# srcDir = '/home/sean/Documents/Forensics/Face2Face/c23/original_images/'
# dstDir = '/home/sean/Documents/Forensics/Face2Face/c23/images/'
# srcDir = '/home/sean/Documents/Forensics/Deepfakes/c23/original_images/'
# dstDir = '/home/sean/Documents/Forensics/Deepfakes/c23/images/'
# srcDir = '/home/sean/Documents/Forensics/FaceSwap/c23/original_images/'
# dstDir = '/home/sean/Documents/Forensics/FaceSwap/c23/images/'
srcDir = '/home/sean/Documents/Forensics/NeuralTextures/c23/original_images/'
dstDir = '/home/sean/Documents/Forensics/NeuralTextures/c23/images/'

print("[Info] Loading directory...")
images = list(paths.list_images(srcDir))
imageNum = len(images)
print("[Info] Loaded {} files.".format(imageNum))

if __name__ == '__main__':
    
    failed = 0
    for i in range(imageNum):
        srcPath = images[i]
        dstPath = srcPath.replace('original_images', 'images')
        # print('src: {}'.format(srcPath))
        # print('dst: {}'.format(dstPath))

        dstFolder = os.path.dirname(dstPath)
        if not os.path.exists(dstFolder):
            os.makedirs(dstFolder)

        try:
            image = cv2.imread(srcPath)
            aligned_image, _ = Align(image)
            cv2.imwrite(dstPath, aligned_image)

            print('[Info] #{}/{} {}'.format(i, imageNum, dstPath))
        except:
            failed = failed + 1
            continue
    
    print('[Info] Finish! Failed: {}'.format(failed))

