# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:02:44 2020

@author: sean8
"""

import cv2
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from imutils import paths
from multiprocessing import Pool

import config
from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm
from align_faces import warp_and_crop_face, get_reference_facial_points

print("[Listing]")
imagePaths = list(paths.list_images(config.srcDir))

cudnn.benchmark = True
torch.set_grad_enabled(False)

# model
model = None
if config.network == "mobile0.25":
    model = cfg_mnet
elif config.network == "resnet50":
    model = cfg_re50


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    #print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    #print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def GetFacialPoints(img_raw):
    img = np.float32(img_raw)
    height, width, _ = img_raw.shape
    scale = torch.Tensor([width, height, width, height])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = net(img)  # forward pass

    priorbox = PriorBox(model, image_size=(height, width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, model['variance'])
    boxes = boxes * scale / config.resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(
        0), prior_data, model['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / config.resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > config.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:config.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
        np.float32, copy=False)
    keep = py_cpu_nms(dets, config.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:config.keep_top_k, :]
    landms = landms[:config.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    torch.cuda.empty_cache()
    return dets


def GetRetinaROI(image, confidence):
    faceConfidence = 0.0
    bestFaceConfidence = 0.0
    bestFace = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    try:
        # search every face in the image
        faces = GetFacialPoints(image)

        for i in range(len(faces)):
            # Each bounding box
            faceConfidence = faces[i][4]
            # print(faceConfidence)

            if len(faces[i]) != 15:
                return None

            if faceConfidence < confidence:
                continue

            if faceConfidence >= bestFaceConfidence:
                bestFaceConfidence = faceConfidence
                bestFace = faces[i]

        # # Show image
        # b = bestFace
        # cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        # cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)
        # cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
        # cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4)
        # cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
        # cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)

        # cv2.imshow("image", image)

        return bestFace
    except:
        return None


def Align(index):
    imgPath = imagePaths[index]
    savePath = config.dstDir + imgPath[21:]

    if os.path.exists(savePath):
        print('[INFO]Existed: #' + str(index))
        return

    image = cv2.imread(imgPath)

    # Parameters for alignment
    confidence = 0.6
    inner_padding_factor = 0.05
    outer_padding = (0, 0)

    try:
        # retinaROI
        retina = GetRetinaROI(image, confidence)
        # get the 5 landmarks of face
        facialPoints = [retina[5], retina[7], retina[9], retina[11], retina[13],
                        retina[6], retina[8], retina[10], retina[12], retina[14]]
        facialPoints = np.reshape(facialPoints, (2, 5))
        # print(facialPoints)
    except:
        return image

    # get the reference 5 landmarks position in the crop settings
    referencePoint = get_reference_facial_points(
        config.output_size, inner_padding_factor, outer_padding, config.default_square)
    # print(referencePoint)

    # alignment
    alignedImg = warp_and_crop_face(
        image, facialPoints, reference_pts=referencePoint, crop_size=config.output_size)

    dirName = os.path.dirname(savePath)
    if not os.path.isdir(dirName):
        os.makedirs(dirName)

    cv2.imwrite(savePath, alignedImg)
    print('[INFO]Save the file: #' + str(index) + '\t' + savePath)

    return alignedImg


# net
net = RetinaFace(cfg=model, phase='test')
net = load_model(net, config.trained_model, config.cpu)
net.eval()

device = torch.device("cpu" if config.cpu else "cuda:0")
net = net.to(device)

if __name__ == '__main__':
    num_img = len(imagePaths)
    print('[INFO]Total files: ' + str(num_img))

    # for i in range(num_img):
    #     Align(i)

    p = Pool(processes=2)
    p.map(Align, range(num_img))
    p.close()
