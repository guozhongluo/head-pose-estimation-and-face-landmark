#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import os
import numpy as np
import cv2
import caffe

def retifyxxyy(img, xxyy):
    """
    let xxyy within image size
    img: image
    xxyy: left, right, top, bottom
    return modified xxyy
    """

    img_height, img_width = img.shape[:2]
    xxyy = retifyxxyysize(img_height, img_width, xxyy)
    return xxyy

def retifyxxyysize(img_height, img_width, xxyy):
    """return xxyy within image region
    img_height:
    img_width:
    xxyy:
    return xxyy
    """

    xxyy[0] = max(xxyy[0], 0)
    xxyy[1] = max(xxyy[1], 0)
    xxyy[2] = max(xxyy[2], 0)
    xxyy[3] = max(xxyy[3], 0)
    xxyy[0] = min(xxyy[0], img_width)
    xxyy[1] = min(xxyy[1], img_width)
    xxyy[2] = min(xxyy[2], img_height)
    xxyy[3] = min(xxyy[3], img_height)
    return xxyy

def getCutSize(xxyy, left, right, top, bottom):   #left, right, top, and bottom
    u"""
    xxyy:
    left:
    right:
    top:
    bottom:
    left, right, top, bottom are ratio.
    The return value is a region with a margin.
    """

    box_width = xxyy[1] - xxyy[0]
    box_height = xxyy[3] - xxyy[2]
    cut_size = np.zeros((4))
    cut_size[0] = xxyy[0] + left * box_width
    cut_size[1] = xxyy[1] + (right - 1) * box_width
    cut_size[2] = xxyy[2] + top * box_height
    cut_size[3] = xxyy[3] + (bottom-1) * box_height
    return cut_size


def dets2xxyys(dets):
    """
    In this module
    xxyy = [left, right, top, bottom]
    """
    xxyys = np.zeros((len(dets), 4))
    for i, d in enumerate(dets):
        xxyys[i, 0] = d.left()
        xxyys[i, 1] = d.right()
        xxyys[i, 2] = d.top()
        xxyys[i, 3] = d.bottom()
    return xxyys


class FacePosePredictor(object):
    """
    A face Pose Predcitor using pre-trained caffe model.

    The orignal code was modified to class version.

    https://github.com/guozhongluo/head-pose-estimation-and-face-landmark

Example:

    posePredictor = facePose.FacePosePredictor()
    predictpoints, landmarks, headposes = posePredictor.predict(frameCopy, np.array([[left, right, top, bottom]]))


    """

    def __init__(self):
        self.M_left = -0.15
        self.M_right = +1.15
        self.M_top = -0.10
        self.M_bottom = +1.25

        self.vgg_height = 224
        self.vgg_width = 224

        vgg_point_MODEL_FILE = 'model/deploy.prototxt'
        vgg_point_PRETRAINED = 'model/68point_dlib_with_pose.caffemodel'
        mean_filename = 'model/VGG_mean.binaryproto'
        self.vgg_point_net = caffe.Net(vgg_point_MODEL_FILE, vgg_point_PRETRAINED, caffe.TEST)
        caffe.set_mode_cpu()
        # caffe.set_mode_gpu()
        # caffe.set_device(0)
        proto_data = open(mean_filename, "rb").read()
        a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
        self.mean = caffe.io.blobproto_to_array(a)[0]


    def predict(self, colorImage, xxyys):
        """
        predcit pitch yaw, roll for each rectangle.
        colorImage:
        xxyys: list of rectangle

        return
        predictpoints: 68 point
        landmarks:
        predictposes: pitch yaw roll
        """


        def getRGBTestPart(img, xxyy, left, right, top, bottom, asHeight, asWidth):
            """return face image as float32
            returned image size width, height

            """

            largexxyy = getCutSize(xxyy, left, right, top, bottom)
            retixxyy = retifyxxyy(img, largexxyy)
            retixxyy = [int(round(x)) for x in retixxyy]
            face = img[retixxyy[2]:retixxyy[3], retixxyy[0]:retixxyy[1], :]
            face = cv2.resize(face, (asHeight, asWidth), interpolation=cv2.INTER_AREA)
            face = face.astype('float32')
            return face


        pointNum = 68

        faceNum = xxyys.shape[0]
        faces = np.zeros((1, 3, self.vgg_height, self.vgg_width))
        predictpoints = np.zeros((faceNum, pointNum*2))
        predictposes = np.zeros((faceNum, 3))
        imgsize = colorImage.shape[:2]
        TotalSize = np.zeros((faceNum, 2))
        normalface = np.zeros(self.mean.shape)

        for i in range(0, faceNum):
            TotalSize[i] = imgsize
            colorface = getRGBTestPart(colorImage, xxyys[i], self.M_left, self.M_right, self.M_top, self.M_bottom, self.vgg_height, self.vgg_width)

            normalface[0] = colorface[:, :, 0]
            normalface[1] = colorface[:, :, 1]
            normalface[2] = colorface[:, :, 2]
            normalface = normalface - self.mean
            faces[0] = normalface

            data4DL = np.zeros([faces.shape[0], 1, 1, 1])
            self.vgg_point_net.set_input_arrays(faces.astype(np.float32), data4DL.astype(np.float32))
            self.vgg_point_net.forward()
            predictpoints[i] = self.vgg_point_net.blobs['68point'].data[0]

            predictposes[i] = 50 * self.vgg_point_net.blobs['poselayer'].data

        predictpoints = predictpoints * self.vgg_height/2 + self.vgg_width/2
        landmarks = self.batchRecoverPart(predictpoints, xxyys, TotalSize, self.M_left, self.M_right, self.M_top, self.M_bottom, self.vgg_height, self.vgg_width)

        return predictpoints, landmarks, predictposes

    def batchRecoverPart(self, predictPoint, totalxxyy, totalSize, left, right, top, bottom, height, width):

        def recover_coordinate(largetxxyy, landmarks, width, height):
            point = np.zeros(np.shape(landmarks))
            cut_width = largetxxyy[1] - largetxxyy[0]
            cut_height = largetxxyy[3] - largetxxyy[2]
            scale_x = cut_width*1.0/width
            scale_y = cut_height*1.0/height
            point[0::2] = [float(j * scale_x + largetxxyy[0]) for j in landmarks[0::2]]
            point[1::2] = [float(j * scale_y + largetxxyy[2]) for j in landmarks[1::2]]
            return point

        def recoverPart(point, xxyy, left, right, top, bottom, img_height, img_width, height, width):
            largexxyy = getCutSize(xxyy, left, right, top, bottom)
            retixxyy = retifyxxyysize(img_height, img_width, largexxyy)
            recover = recover_coordinate(retixxyy, point, height, width)
            recover = recover.astype('float32')
            return recover

        recoverPoint = np.zeros(predictPoint.shape)
        for i in range(0, predictPoint.shape[0]):
            recoverPoint[i] = recoverPart(predictPoint[i], totalxxyy[i], left, right, top, bottom, totalSize[i, 0], totalSize[i, 1], height, width)
        return recoverPoint


    def predict1(self, colorImage, xxyy):
        """
        predcit pitch yaw, roll for single rectangle.
        colorImage:
        xxyy: single rectangle

        return value
        predictposes[0, :] : pitch, yaw, roll

        """
        predictpoints, landmarks, predictposes = self.predict(colorImage, np.array([xxyy]))

        return predictpoints[0], landmarks[0], predictposes[0, :]


def roundByD(angle, delta):
    """round angle by delta
    angle:
    delta:
>>> roundByD(8, 10)
10.0
>>> roundByD(-9.5, 10)
-10.0
    """
    return delta*round(angle/float(delta))


def getPyrStr(pitch, yaw, roll):
    """
    pitch:
    yaw:
    roll:
    """

    pitchDelta = 5
    yawDelta = 5
    rollDelta = 10

    pyrDir = "P_%+03d_Y_%+03d_R_%+03d" % (roundByD(pitch, pitchDelta), roundByD(yaw, yawDelta), roundByD(roll, rollDelta))
    return pyrDir

def getPyStr(pitch, yaw):
    """
    pitch:
    yaw:
    """

    pitchDelta = 5
    yawDelta = 5
    rollDelta = 10

    pyrDir = "P_%+03d_Y_%+03d" % (roundByD(pitch, pitchDelta), roundByD(yaw, yawDelta))
    return pyrDir

def getPyrDir(outDir, pitch, yaw, roll):
    """
    pitch:
    yaw:
    roll:
    """

    pyrDir = os.path.join(outDir, getPyrStr(pitch, yaw, roll))
    if not os.path.isdir(pyrDir):
        os.makedirs(pyrDir)
    return pyrDir

def getPyDir(outDir, pitch, yaw):
    """
    pitch:
    yaw:
    roll:
    """

    pitchDelta = 5
    yawDelta = 5
    rollDelta = 10

    pyrDir = "P_%+03d_Y_%+03d" % (roundByD(pitch, pitchDelta), roundByD(yaw, yawDelta))
    pyrDir = os.path.join(outDir, pyrDir)
    if not os.path.isdir(pyrDir):
        os.makedirs(pyrDir)
    return pyrDir
