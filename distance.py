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

boxes = []
objid = 0

class Box:
    def __init__(self, top, left, bot, right, label):
        self.top = top
        self.left = left
        self.bot = bot
        self.right = right
        self.label = label
        self.id = 0
        self.distance = 0

    def calcDistance(self):
        camdimH = 2.67
        fl = 3200
        height = 1.0
        if self.label == 'car':
            dh = 1.6
        elif self.label == 'ped': 
            dh = 1.7
        else:
            dh = 0.5
        self.distance = ((fl * dh * height) / camdimH) / (self.bot - self.top)

def readjson(filename):
    frames = []
    with open(filename) as file:
        content = json.load(file)
    return content

def main():
    global boxes, objid
    maxrange = 100
    maxwidth = 500
    cam = video.create_capture('cut1.mp4')
    content = readjson('video.json')

    for frames in content:
        boxes = []
        ret, frame = cam.read()
        frameheight = np.size(frame, 0)
        birdframe = np.zeros((frameheight, maxwidth, 3), np.uint8)
        birdframe[:] = (255, 255, 255)
        for obj in frames:
            if obj['confidence'] >= 0.3:
                newbox = Box(obj['topleft']['y'], obj['topleft']['x'], obj['bottomright']['y'], 
                    obj['bottomright']['x'], obj['label']) 
                boxes.append(newbox)
                newbox.calcDistance()
        for box in boxes:
            cv2.rectangle(frame, (box.left, box.top), (box.right, box.bot), (255, 0, 0), 2)
            draw_str(frame, (box.left, box.top-10), str(box.distance)[0:5])
            cv2.rectangle(birdframe, (int(box.left * maxwidth / 1280), 
                int(frameheight - box.distance * 720 / maxrange)), 
                (int(box.right * maxwidth / 1280), 
                int(frameheight - box.distance * 720 / maxrange + 10)), 
                (255, 0, 0), 2)
        cv2.imshow('track', frame)
        cv2.imshow('birdseye view', birdframe)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
