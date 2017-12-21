# -*- coding: utf-8 -*-
# pylint: disable=C0103
#usage :python landmarkPredict.py predictVideo  testList.txt

import numpy as np
import cv2
import caffe

def retifyBBox(img, bbox):
    img_height, img_width = img.shape[:2]
    bbox = retifyBBoxSize(img_height, img_width, bbox)
    return bbox

def retifyBBoxSize(img_height, img_width, bbox):
    """return bbox within image region
    img_height:
    img_width:
    bbox:
    """

    bbox[0] = max(bbox[0], 0)
    bbox[1] = max(bbox[1], 0)
    bbox[2] = max(bbox[2], 0)
    bbox[3] = max(bbox[3], 0)
    bbox[0] = min(bbox[0], img_width)
    bbox[1] = min(bbox[1], img_width)
    bbox[2] = min(bbox[2], img_height)
    bbox[3] = min(bbox[3], img_height)
    return bbox

def getCutSize(bbox, left, right, top, bottom):   #left, right, top, and bottom
    r"""
    left, right, top, bottom は比率
    戻り値はマージン付きの領域になる。
    """

    box_width = bbox[1] - bbox[0]
    box_height = bbox[3] - bbox[2]
    cut_size = np.zeros((4))
    cut_size[0] = bbox[0] + left * box_width
    cut_size[1] = bbox[1] + (right - 1) * box_width
    cut_size[2] = bbox[2] + top * box_height
    cut_size[3] = bbox[3] + (bottom-1) * box_height
    return cut_size




def dets2bboxs(dets):
    """
    In this module
    bbox = [left, right, top, bottom]
    """
    bboxs = np.zeros((len(dets), 4))
    for i, d in enumerate(dets):
        bboxs[i, 0] = d.left();
        bboxs[i, 1] = d.right();
        bboxs[i, 2] = d.top();
        bboxs[i, 3] = d.bottom();
    return bboxs;


class FacePosePredictor(object):

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


    def predict(self, colorImage, bboxs):


        def getRGBTestPart(img, bbox, left, right, top, bottom, asHeight, asWidth):
            """return face image as float32
            returned image size width, height

            """

            largeBBox = getCutSize(bbox, left, right, top, bottom)
            retiBBox = retifyBBox(img, largeBBox)
            retiBBox = [int(round(x)) for x in retiBBox]
            face = img[retiBBox[2]:retiBBox[3], retiBBox[0]:retiBBox[1], :]
            face = cv2.resize(face, (asHeight, asWidth), interpolation=cv2.INTER_AREA)
            face = face.astype('float32')
            return face


        pointNum = 68

        faceNum = bboxs.shape[0]
        faces = np.zeros((1, 3, self.vgg_height, self.vgg_width))
        predictpoints = np.zeros((faceNum, pointNum*2))
        predictpose = np.zeros((faceNum, 3))
        imgsize = colorImage.shape[:2]
        TotalSize = np.zeros((faceNum, 2))
        normalface = np.zeros(self.mean.shape)

        for i in range(0, faceNum):
            TotalSize[i] = imgsize
            colorface = getRGBTestPart(colorImage, bboxs[i], self.M_left, self.M_right, self.M_top, self.M_bottom, self.vgg_height, self.vgg_width)

            normalface[0] = colorface[:, :, 0]
            normalface[1] = colorface[:, :, 1]
            normalface[2] = colorface[:, :, 2]
            normalface = normalface - self.mean
            faces[0] = normalface

            data4DL = np.zeros([faces.shape[0], 1, 1, 1])
            self.vgg_point_net.set_input_arrays(faces.astype(np.float32), data4DL.astype(np.float32))
            self.vgg_point_net.forward()
            predictpoints[i] = self.vgg_point_net.blobs['68point'].data[0]

            predictpose[i] = 50 * self.vgg_point_net.blobs['poselayer'].data

        predictpoints = predictpoints * self.vgg_height/2 + self.vgg_width/2
        level1Point = self.batchRecoverPart(predictpoints, bboxs, TotalSize, self.M_left, self.M_right, self.M_top, self.M_bottom, self.vgg_height, self.vgg_width)

        return predictpoints, level1Point, predictpose

    def batchRecoverPart(self, predictPoint, totalBBox, totalSize, left, right, top, bottom, height, width):

        def recover_coordinate(largetBBox, facepoint, width, height):
            point = np.zeros(np.shape(facepoint))
            cut_width = largetBBox[1] - largetBBox[0]
            cut_height = largetBBox[3] - largetBBox[2]
            scale_x = cut_width*1.0/width;
            scale_y = cut_height*1.0/height;
            point[0::2] = [float(j * scale_x + largetBBox[0]) for j in facepoint[0::2]]
            point[1::2] = [float(j * scale_y + largetBBox[2]) for j in facepoint[1::2]]
            return point

        def recoverPart(point, bbox, left, right, top, bottom, img_height, img_width, height, width):
            largeBBox = getCutSize(bbox, left, right, top, bottom)
            retiBBox = retifyBBoxSize(img_height, img_width, largeBBox)
            recover = recover_coordinate(retiBBox, point, height, width)
            recover = recover.astype('float32')
            return recover

        recoverPoint = np.zeros(predictPoint.shape)
        for i in range(0, predictPoint.shape[0]):
            recoverPoint[i] = recoverPart(predictPoint[i], totalBBox[i], left, right, top, bottom, totalSize[i, 0], totalSize[i, 1], height, width)
        return recoverPoint

