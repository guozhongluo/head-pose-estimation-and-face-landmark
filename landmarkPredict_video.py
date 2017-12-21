#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
#usage :python landmarkPredict.py predictVideo  testList.txt

import os
import sys
import time
import cv2
import dlib # http://dlib.net

#from  facePose import *
import facePose 

"""
In this module
bbox = [left, right, top, bottom]
"""

pose_name = ['Pitch', 'Yaw', 'Roll']     # respect to  ['head down','out of plane left','in plane right']

outDir = os.path.expanduser("~/output")

def roundByD(angle, delta):
    """
    angle:
    delta:
>>> roundByD(8, 10)
10.0
>>> roundByD(-9.5, 10)
-10.0
    """
    return delta*round(angle/float(delta))
    
    

def show_image(img, facepoint, bboxs, headpose):
    u"""
    img:
    facepoint: landmark points
    bboxs: dlibの顔検出枠を bounding box としたもののリスト
    headpose:
        headpose[0, :]: 0番目の顔のpitch, yaw, row 
    """

    orgImg = img+0

    pitchDelta = 10
    yawDelta = 10
    rollDelta = 10

    system_height = 650
    system_width = 1280


    for faceNum in range(0, facepoint.shape[0]):
        cv2.rectangle(img, (int(bboxs[faceNum, 0]), int(bboxs[faceNum, 2])), (int(bboxs[faceNum, 1]), int(bboxs[faceNum, 3])), (0, 0, 255), 2)
        for p in range(0, 3):

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, '{:s} {:.2f}'.format(pose_name[p], headpose[faceNum, p]), (10, 400+25*p), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(orgImg, '{:s} {:.2f}'.format(pose_name[p], headpose[faceNum, p]), (10, 400+25*p), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        for i in range(0, facepoint.shape[1]/2):
            cv2.circle(img, (int(round(facepoint[faceNum, i*2])), int(round(facepoint[faceNum, i*2+1]))), 1, (0, 255, 0), 2)
        pitch = headpose[faceNum, 0]
        yaw = headpose[faceNum, 1]
        roll = headpose[faceNum, 2]

        pyrDir = "P_%02d_Y_%02d_R_%02d" % (roundByD(pitch, pitchDelta), roundByD(yaw, yawDelta), roundByD(roll, rollDelta))
        pyrDir = os.path.join(outDir, pyrDir)
        if not os.path.isdir(pyrDir):
            os.makedirs(pyrDir)

        datetimeStr = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        pngname = os.path.join(pyrDir, "%s.jpg" % datetimeStr)
        cv2.imwrite(pngname, orgImg)

    if facepoint.shape[0] < 1:
        pyrDir = "couldNotDetect"
        pyrDir = os.path.join(outDir, pyrDir)
        if not os.path.isdir(pyrDir):
            os.makedirs(pyrDir)

        datetimeStr = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        pngname = os.path.join(pyrDir, "%s.jpg" % datetimeStr)
        cv2.imwrite(pngname, orgImg)
        print pngname


    height, width = img.shape[:2]
    if height > system_height or width > system_width:
        height_radius = system_height*1.0/height
        width_radius = system_width*1.0/width
        radius = min(height_radius, width_radius)
        img = cv2.resize(img, (0, 0), fx=radius, fy=radius)

    cv2.imshow("img", img)



def predictVideo(uvcID):
    """
    uvcID: video camera ID
    """

    detector = dlib.get_frontal_face_detector()
    posePredictor = facePose.FacePosePredictor()

    cap = cv2.VideoCapture(uvcID)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)

    while True:
        ok, colorImage = cap.read()
        if not ok:
            continue

        numUpSampling = 0
        dets, scores, idx = detector.run(colorImage, numUpSampling)
        bboxs = facePose.dets2bboxs(dets)

        predictpoints, facepoint, predictpose = posePredictor.predict(colorImage, bboxs)

        show_image(colorImage, facepoint, bboxs, predictpose)

        k = cv2.waitKey(10) & 0xff
        if k == ord('q') or k == 27:
            break

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        exit()

    uvcID = int(sys.argv[1])
    predictVideo(uvcID)

