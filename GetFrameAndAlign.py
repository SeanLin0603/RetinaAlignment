# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 09:39:40 2020

@author: sean8
"""

import os
import cv2
from pathlib import Path
from multiprocessing import Pool
# import torch
# torch.multiprocessing.set_start_method('spawn')


from align_main import Align

# srcDir = 'C:\\Users\\sean8\\Desktop\\SIWM dataset\\Videos\\'
# srcDir = 'C:\\Users\\SeanVIP\\Documents\\FaceForensic\\'
# dstDir = 'C:\\Users\\SeanVIP\\Documents\\frame\\'

srcDir = '/home/sean/Documents/faceforensic/'
dstDir = '/home/sean/Documents/faceforensic/frame/'
offset = len(srcDir)

print("[Listing]")
videolist = list(Path(srcDir).rglob('*.avi'))
print("[Listed]")


def getFrame(index):
    global videolist
    path = str(videolist[index])

    # tokens = path.split('/')
    # saveDir = '/'.join(tokens)
    
    saveDir = dstDir + path[offset:-4]

    if not os.path.isdir(saveDir):
        print(saveDir)
        os.makedirs(saveDir, mode=755)

    # Opens the Video file
    video = cv2.VideoCapture(path)
    i = 0
    while(video.isOpened()):
        ret, frame = video.read()
        if ret == False:
            break

        try:
            frame = Align(frame)
        except:
            continue

        saveName = os.path.join(saveDir, str(i) + '.jpg')
        print('[Info] SaveName: {}'.format(saveName))
        cv2.imwrite(saveName, frame)
        i += 1

    video.release()
    print('[Info] Finished video: {}\n'.format(path))


if __name__ == '__main__':
    
    length = len(videolist)
    print("[Info] length: {}".format(length))

    for i in range(1412, length):
        print("[Info] Index: {}/{}".format(i, length))
        getFrame(i)

    # p = Pool(processes=20)
    # p.map(getFrame, range(len(videolist)))
    # p.close()
