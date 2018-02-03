#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
import numpy as np
import PIL.Image # https://github.com/python-pillow/Pillow

def largestRect(rects):
    u"""retturn largest rect in rects
    rects: list of rect
    """

    if len(rects) < 2:
        return rects

    largest = rects[0]
    for i in range(1, len(rects)):
        if rects[i][2] > largest[2]:
            largest = rects[i]

    return largest

def overlapRange(lim1, lim2):
    """return overlapped lim
    lim1:
    lim2:
    """

    start = max(lim1[0], lim2[0])
    stop = min(lim1[1], lim2[1])

    if start > stop:
        return [None, None]
    else:
        return [start, stop]

def overlapRectArea(rect1, rect2):
    """return overlapped area
    rect1:
    rect2:
    """

    left1, right1 = rect1[0], rect1[0]+rect1[2]
    top1, bottom1 = rect1[1], rect1[1]+rect1[3]


    left2, right2 = rect2[0], rect2[0]+rect2[2]
    top2, bottom2 = rect2[1], rect2[1]+rect2[3]

    [left3, right3] = overlapRange([left1, right1], [left2, right2])
    [top3, bottom3] = overlapRange([top1, bottom1], [top2, bottom2])

    if None in (left3, top3, right3, bottom3):
        return 0.0
    else:
        area = (right3-left3)*(bottom3-top3)
        area >= 0.0
        return area

def getIoU(rect1, rect2):
    u"""
    return intersection  over union
"""

    area1 = rect1[2]*rect1[3]
    area2 = rect2[2]*rect2[3]
    intersection = overlapRectArea(rect1, rect2)
    assert intersection >= 0
    union = area1+area2 - intersection
    assert union >= 0

    IoU = intersection/float(union)
    assert IoU >= 0
    return IoU


def rect2bbox(rect):
    """convert rect into bbox.
    tracker.init() need this data type.
    """

    assert len(rect) == 4
    x, y, w, h = rect
    assert w > 0
    assert h > 0
    return (long(x), long(y), long(w), long(h))


def dets2rects(dets):
    """
    convert dets type to rect type.
    left, top, width, height
"""

    rects = [[d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top()] for d in dets]
    return rects


def getBestIoU(rects, states):
    u"""find best matched tracking for each rect.
    rects: detected rects
    states: tracking states

    """

    asTrack = len(rects)*[None]
    alreadyFounds = len(rects)*[0.0]

    for j, rect in enumerate(rects):# 検出について
        for k, (_, bbox) in  enumerate(states):#追跡について
            IoU = getIoU(bbox, rect)
            assert IoU >= 0.0
            assert len(rect) == 4
            assert rect[2] > 0
            assert rect[3] > 0
            if IoU > alreadyFounds[j]:
                alreadyFounds[j] = max(alreadyFounds[j], IoU)
                asTrack[j] = k
    return alreadyFounds, asTrack


def expandRegion(rect, rate):
    """expand rectange x,y,w,h keeping center postion.
    rect: x,y,w,h
    rate
    """

    x, y, w, h = rect
    xc, yc = x+w/2, y+w/2

    nw = int(rate*w)
    nh = int(rate*h)

    nx = xc - nw/2
    ny = yc - nh/2
    return [nx, ny, nw, nh]

def sizedCrop(img, xyxy):
    u"""Returns a rectangular region from this alignedImg.
    The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
    When outside of the image is specified, it is displayed as black without an error.
    """

    pilImg = PIL.Image.fromarray(img)
    pilsubImg = pilImg.crop(xyxy)
    subImg = np.asarray(pilsubImg)
    return subImg


def test_overlapRegion():
    lim = overlapRange([0, 10], [0, 10])
    assert lim == [0, 10]
    lim = overlapRange([0, 10], [0, 20])
    assert lim == [0, 10]
    lim = overlapRange([0, 10], [-10, 20])
    assert lim == [0, 10]


    lim = overlapRange([0, 10], [5, 10])
    assert lim == [5, 10]

    lim = overlapRange([0, 10], [5, 20])
    assert lim == [5, 10]

    lim = overlapRange([-10, 10], [5, 20])
    assert lim == [5, 10]


    lim = overlapRange([5, 10], [5, 20])
    assert lim == [5, 10]


def test_getIoU():
    IoU = getIoU([10, 20, 30, 40], [10, 20, 30, 40])
    print IoU
    assert IoU == 1.0

    IoU = getIoU([10, 20, 30, 40], [10, 20, 30, 20])
    print IoU
    assert IoU <= 0.5+0.01
    assert 0.5 - 0.01 <= IoU

    IoU = getIoU([10, 20, 30, 40], [10, 25, 30, 40])
    print IoU
    assert IoU < 1.0
    assert IoU >= 0.0


if __name__ == "__main__":
    test_overlapRegion()
    test_getIoU()