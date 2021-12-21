# -*- coding: utf-8 -*-
import cv2
import numpy as np
from numpy.core.fromnumeric import nonzero
import torch
import torch.backends.cudnn as cudnn
torch.multiprocessing.set_start_method('spawn')

from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace, load_model
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm
from align_faces import warp_and_crop_face, get_reference_facial_points
import config

cudnn.benchmark = True
torch.set_grad_enabled(False)

cfg = None
if config.network == "mobile0.25":
    cfg = cfg_mnet
elif config.network == "resnet50":
    cfg = cfg_re50

# net and model
net = RetinaFace(cfg=cfg, phase='test')
net = load_model(net, config.pretrain_weight, config.gpu)
net.eval()

device = torch.device("cuda:0" if config.gpu else "cpu")
net = net.to(device)

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

    priorbox = PriorBox(cfg, image_size=(height, width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / config.resize
    boxes = boxes.cpu().detach().numpy()
    scores = conf.squeeze(0).data.cpu().detach().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / config.resize
    landms = landms.cpu().detach().numpy()

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
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:config.keep_top_k, :]
    landms = landms[:config.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    torch.cuda.empty_cache()
    return dets

def GetRetinaROI(image):
    bestConfidence = 0.0
    bestLandmark = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # get face landmark
    faces = GetFacialPoints(image)
    # print("[Info] FaceNum: {}, {}".format(len(faces), faces))

    # Each face
    for face in faces:
        if len(face) != 15:
            return None

        faceConfidence = face[4]
        # print("[Info] FaceConfidence: {}".format(faceConfidence))

        if faceConfidence < config.confidence:
            continue

        if faceConfidence >= bestConfidence:
            bestConfidence = faceConfidence
            bestLandmark = face
    return bestLandmark

def draw(image, landmark):
    thickness = 4
    radius = 1
    landmark = landmark.astype(int)

    cv2.rectangle(image, (landmark[0], landmark[1]), (landmark[2], landmark[3]), (0, 255, 0), 2)
    cv2.circle(image, (landmark[5], landmark[6]), radius, (0, 0, 255), thickness)
    cv2.circle(image, (landmark[7], landmark[8]), radius, (0, 255, 255), thickness)
    cv2.circle(image, (landmark[9], landmark[10]), radius, (255, 0, 255), thickness)
    cv2.circle(image, (landmark[11], landmark[12]), radius, (0, 255, 0), thickness)
    cv2.circle(image, (landmark[13], landmark[14]), radius, (255, 0, 0), thickness)
    return image

def Align(image):
    # print('[Info] Input image size: {}'.format(image.shape))
    landmark = GetRetinaROI(image)
    
    # detected position
    facialPoints = [landmark[5], landmark[7], landmark[9], landmark[11], landmark[13],
                    landmark[6], landmark[8], landmark[10], landmark[12], landmark[14]]
    facialPoints = np.reshape(facialPoints, (2, 5))

    # ideal position
    referencePoint = get_reference_facial_points(
        config.output_size, config.inner_padding_factor, config.outer_padding, config.default_square)

    alignedImg = warp_and_crop_face(
        image, facialPoints, reference_pts=referencePoint, crop_size=config.output_size)

    # draw landmark
    if config.draw_landmark:
        drawImg = draw(image, landmark)
    else:
        drawImg = None

    # return alignedImg, image
    return alignedImg, drawImg

