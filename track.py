#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

import numpy as np
import cv2
import video
from common import anorm2, draw_str
import json

objlist = ['car', 'ped', 'stop']
tracks = []
points = []
boxes = []
prev_gray = None
frame_idx = 0
track_len = 5
detect_interval = 1
objid = 0
max_history = 24
prediction_rate = 1
prediction_range_begin = 15

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.numDetected = 0
        self.detected = False
        self.findInBox()
        self.track = []

    def printPointList(self, ptlist):
        temp = ''
        for p in ptlist:
            temp += '(' + str(p.x) + ', ' + str(p.y) + '), '
        print temp

    def setTrack(self, track):
        self.track = track

    def findInBox(self):
        for box in boxes:
            if self.x > box.left and self.x < box.right and self.y > box.top and self.y < box.bot:
                self.detected = True
                self.numDetected += 1

class Box:
    def __init__(self, top, left, bot, right, label):
        self.top = top
        self.left = left
        self.bot = bot
        self.right = right
        self.label = label
        self.id = 0
        self.pts = []
        self.confidence = 0
        self.history = []
        self.color = self.setColor()

    def setColor(self):
        if self.label == 'car':
            return (96, 204, 255)
        elif self.label == 'ped':
            return (212,175,55)
        elif self.label == 'stop':
            return (255, 204, 204)

    def addHistory(self):
        global max_history
        while len(self.history) >= max_history:
            self.history.pop(0)
        newBox = Box(self.top, self.left, self.bot, self.right, self.label)
        self.history.append(newBox)
        return self.history

    def findFeatures(self):
        featuresfound = False
        for pt in points:
            if pt.x > self.left and pt.x < self.right and pt.y > self.top and pt.y < self.bot:
                self.pts.append(pt)
                featuresfound = True

    def findMaxID(self, idlist):
        curmax = 0
        curmaxk = -1
        for k, v in idlist.iteritems():
            if v > curmax:
                curmax = v
                curmaxk = k
        return k

    def matchPoints(self, prevboxes):
        global objid
        boxlist = {}
        historylist = {}
        for b in prevboxes:
            if b.label == self.label:
                for pt in self.pts:
                    for p in pt.track:
                        if p in b.pts:
                            if str(b.id) in boxlist.keys():
                                boxlist[str(b.id)] += 1
                            else:
                                boxlist[str(b.id)] = 1
                                historylist[str(b.id)] = b.addHistory()
        self.calcConfidenceLevel(boxlist)
        if len(boxlist) > 0:
            self.id = self.findMaxID(boxlist)
            self.history = historylist[self.id]
            return
        self.id = objid
        objid += 1

    def calcConfidenceLevel(self, boxlist):
        conf = 0
        numboxes = len(boxlist)
        numpoints = len(self.pts)
        if numboxes == 0:
            conf += 0.1
        elif numboxes == 1:
            conf += 0.3
        elif numboxes > 1:
            conf -= (numboxes * 0.3)
        for k, v in boxlist.iteritems():
            conf += (v * 0.18)/25
            break
        conf += (numpoints * 0.12)/35
        self.confidence = conf

def optflow(frame):
    global tracks, points, frame_idx, prev_gray

    points = []

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()
    newimage = np.zeros((np.size(frame, 0), np.size(frame, 1), 3), np.uint8)
    newimage[:] = (0, 0, 0)

    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([(tr[-1].x, tr[-1].y) for tr in tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            newpt = Point(x, y)
            if newpt.detected == True:
                tr.append(newpt)
                points.append(newpt)
                newpt.setTrack(tr)
                if len(tr) > track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

        tracks = new_tracks
        temp1 = []
        for paths in tracks:
            temp2 = []
            for pts in paths[-10:]:
                temp2.append((pts.x, pts.y))
            temp1.append(np.int32(temp2))
        cv2.polylines(vis, temp1, False, (0, 255, 0))
        draw_str(vis, (20, 20), 'track count: %d' % len(tracks))

    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32((tr[-1].x, tr[-1].y)) for tr in tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                newpt = Point(x, y)
                points.append(newpt)
                tracks.append([newpt])

    frame_idx += 1
    prev_gray = frame_gray
    return vis


def readjson(filename):
    frames = []
    with open(filename) as file:
        content = json.load(file)
    return content


def main():
    global boxes, objid

    cam = video.create_capture('cut1.mp4')
    content = readjson('video.json')

    framecount = 0
    for frames in content:
        oldboxes = boxes
        boxes = []
        ret, frame = cam.read()
        for obj in frames:
            if obj['confidence'] >= 0.3 and obj['label'] in objlist:
                newbox = Box(obj['topleft']['y'], obj['topleft']['x'], obj['bottomright']['y'], 
                    obj['bottomright']['x'], obj['label']) 
                if framecount == 0:
                    newbox.id = objid
                    objid += 1
                boxes.append(newbox)
        frame = optflow(frame)
        for box in boxes:
            box.findFeatures()
            box.matchPoints(oldboxes)
            cv2.rectangle(frame, (box.left, box.top), (box.right, box.bot), box.color, 3)
            # if len(box.history) > 0:
            #     center1 = ((box.history[0].left+box.history[0].right)/2, (box.history[0].top+box.history[0].bot)/2)
            #     last = len(box.history) - 1
            #     center2 = ((box.history[last].left+box.history[last].right)/2, (box.history[last].top+box.history[last].bot)/2)
            #     cv2.arrowedLine(frame, center1, center2, (0, 0, 225), 2)
                # for i in range(prediction_range_begin, last):
                #     if i%prediction_rate == 0:
                #         prediction_lefttop = (2*box.history[i].left - box.history[0].left, 2*box.history[i].top - box.history[0].top)
                #         prediction_botright = (2*box.history[i].right - box.history[0].right, 2*box.history[i].bot - box.history[0].bot)
                #         cv2.rectangle(frame, prediction_lefttop, prediction_botright, box.color, 1)
            draw_str(frame, (box.left, box.top-10), str(box.id))
        cv2.imshow('track', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        framecount += 1
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()