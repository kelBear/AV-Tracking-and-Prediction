import numpy as np
import cv2
import video
from common import anorm2, draw_str
import json
import math
from bayesian_network import *
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

objlist = ['car', 'ped', 'stop']
tracks = []
points = []
boxes = []
lostBoxes = []
prev_gray = None
frame_idx = 0
track_len = 5
detect_interval = 1
objid = 0
max_history = 30
prediction_rate = 1
prediction_range_begin = 15
facedist = 64
prediction_modifier = 1

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
        self.prediction = []

    def setColor(self):
        if self.label == 'car':
            return (96, 204, 255)
        elif self.label == 'ped':
            return (212,175,55)
        elif self.label == 'stop':
            return (255, 204, 204)

    def addHistory(self, oldBox):
        global max_history
        #newBox = Box(self.top, self.left, self.bot, self.right, self.label)
        newHistory = oldBox.history
        newHistory.append(self)
        self.history = newHistory
        while len(self.history) >= max_history:
            self.history.pop(0)

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

    def matchPoints(self):
        global objid, boxes, lostBoxes
        #boxlist = optflowMatch(self)

        # self.calcConfidenceLevel(boxlist)
        # if len(boxlist) > 0:
        #     index = int(self.findMaxID(boxlist))
        #     self.id = boxes[index].id
        #     self.addHistory(boxes[index])
        #     removeBox(self.id)
        #     return

        # matchBox = self.directionMatch(optflowMatch(self))
        # if matchBox!=None:
        #     self.id = matchBox.id
        #     self.addHistory(matchBox)
        #     removeBox(self.id)
        #     return
        matchBox = self.directionMatch(centerMatch(self, boxes))
        if matchBox!=None:
            self.id = matchBox.id
            self.addHistory(matchBox)
            removeBox(self.id)
            return

        lostBox = self.directionMatch(predictionMatch(self, lostBoxes))
        if lostBox!=None:
            # print 'reduced'
            # print lostBox.id
            self.id = lostBox.id
            self.addHistory(lostBox)
            removeBox(self.id)
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

    def directionMatch(self, blist):
        if blist == None:
            return None
        if len(blist) < 1:
            return None
        choose =  blist[0]
        minimum = 1000
        for box in blist:
            l = len(box.history) 
            if len(box.history) > prediction_range_begin:
                selfCenter = [(self.left+self.right)/2 , (self.top+self.bot)/2]
                hist1Center = [(box.history[0].left+box.history[0].right)/2 , (box.history[0].top+box.history[0].bot)/2]
                hist2Center = [(box.history[l-1].left+box.history[l-1].right)/2 , (box.history[l-1].top+box.history[l-1].bot)/2]
                # if (hist2Center[1] != hist1Center[1]) and (hist1Center[1] != selfCenter[1]):
                #     slope1 = (hist2Center[0] - hist1Center[0])/(hist2Center[1] - hist1Center[1])
                #     slope2 = (hist1Center[0] - selfCenter[0])/(hist1Center[1] - selfCenter[1])
                #     if slope1 != 0:
                #         c = abs((slope2-slope1)/abs(slope1))
                #         if c < minimum:
                #             minimum = c
                #             choose = box
                histxavg = 0
                histyavg = 0
                for i in range(0, 9):
                    histxavg += (box.history[l-1-i].left+box.history[l-1-i].right)/2
                    histyavg += (box.history[l-1-i].top+box.history[l-1-i].bot)/2
                histxavg /= 9
                histyavg /= 9
                vec1 = [hist2Center[0] - hist1Center[0], hist2Center[1] - hist1Center[1]]
                vec2 = [selfCenter[0] - histxavg, selfCenter[1] - histyavg]
                dot = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                if norm1 > 20:
                    norm2 = np.linalg.norm(vec2)
                    try:
                        angle = math.degrees(math.acos(dot/(norm1*norm2)))
                        if angle < minimum:
                            minimum = angle
                            choose = box
                    except:
                        pass
                    
        if minimum > 110 and minimum < 1000 and box.label == 'ped':
            #print('box: ' + str(box.id) + ', angle: ' + str(angle))
            return None

        return choose


def removeBox(bid):
    global boxes, lostBoxes
    for box in boxes:
        if box.id == bid:
            boxes.remove(box)
    for box in lostBoxes:
        if box.id == bid:
            lostBoxes.remove(box)

def histInTrack(box, hist):
    # any point in history
    for pt in hist.pts:
        # all the points in box tracks
        for pt2 in box.pts:
            for t in pt2.track:               
                if t in hist.pts:      
                    return 1

    return 0

def optflowMatch(box):
    global objid, boxes
    boxlist = {}
    i = 0
    j = 0
    # for b in boxes:
    #     if b.label == box.label:
    #         for pt in box.pts:
    #             for p in pt.track:
    #                 if p in b.pts:
    #                     if str(b.id) in boxlist.keys():
    #                         boxlist[str(i)] += 1
    #                     else:
    #                         boxlist[str(i)] = 1
    #     i = i + 1
    # return boxlist
    maximum = 0
    choose = []
    for oldBox in boxes:
        i = 0
        histCounter = 0
        for hist in oldBox.history:
            i += histInTrack(box, hist)
            histCounter += 1
            if histCounter > track_len:
                break
        c = float(maximum)/float(min(track_len, max_history))
        if c > 0.6:
            choose.append(oldBox)
    # if choose != None:
    #     print ('box: ' + str(choose.id) + ', history found: ' + str(float(maximum)/float(min(track_len, max_history))))
    
    return choose


def simpleBoxMatch(box1, box2):
    x1 = (box1.left + box1.right) / 2
    y1 = (box1.top + box1.bot) / 2
    x2 = (box2.left + box2.right) / 2
    y2 = (box2.top + box2.bot) / 2

    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

def calcAreaOverlap(box1, box2):
    left = max(box1.left, box2.left)
    right = min(box1.right, box2.right)
    bot = min(box1.bot, box2.bot)
    top = max(box1.top, box2.top)
    if left < right and top < bot:
        area1 = (box1.right - box1.left)*(box1.bot - box1.top)
        area2 = (box2.right - box2.left)*(box2.bot - box2.top)
        overlaparea = (right - left)*(bot - top)
        return float(overlaparea)/float(max(area1,area2))
    return 0

def predictionMatch(box, blist):
    minimum = -1
    maximum = 10
    choose = []
    minbox = None
    for lostBox in blist:
        if lostBox.label == box.label:
            for prediction in lostBox.prediction:
                c = simpleBoxMatch(box, prediction)
                a = calcAreaOverlap(box, prediction)
                if c < facedist * prediction_modifier:
                    if c < minimum:
                        minimum = c
                        minbox = prediction
                    choose.append(lostBox)
                    break
                # elif a > 0.75 and (maximum == 1 or a > maximum):
                #     maximum = a
                #     choose = lostBox
    if minbox != None:
        choose.append(choose[0])
        choose[0] = minbox
    return choose

def centerMatch(box, blist):
    minimum = -1
    maximum = 10
    choose = []
    minbox = None
    for oldBox in blist:
        if oldBox.label == box.label:
            c = simpleBoxMatch(box, oldBox)
            a = calcAreaOverlap(box, oldBox)
            if a > 0.2:
                if c < facedist:
                    choose.append(oldBox)
                    if c < minimum:
                        minimum = c
                        minbox = prediction

            # elif a > 0.75 and (maximum == 1 or a > maximum):
            #     maximum = a
            #     choose = lostBox
    if minbox != None:
        choose.append(choose[0])
        choose[0] = minbox
    return choose


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
                #cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

        tracks = new_tracks
        temp1 = []
        for paths in tracks:
            temp2 = []
            for pts in paths[-10:]:
                temp2.append((pts.x, pts.y))
            temp1.append(np.int32(temp2))
        #cv2.polylines(vis, temp1, False, (0, 255, 0))
        #draw_str(vis, (20, 20), 'track count: %d' % len(tracks))

    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        # for x, y in [np.int32((tr[-1].x, tr[-1].y)) for tr in tracks]:
        #     cv2.circle(mask, (x, y), 5, 0, -1)
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

def lineOfBestFit(coords):
    x = coords[0]
    y = coords[1]
    deg = 2
    coeff = numpy.polyfit(x, y, deg, rcond=None, full=False)
    #print (coeff)
    return coeff

def calcBayesianNetwork(frame):
    pointList = [(537, 224), (456, 224), (798, 720), (1280, 720), (1280, 360), (537, 224)]
    poly = geometry.Polygon(pointList)
    cv2.polylines(frame, [np.array(pointList, np.int32).reshape((-1, 1, 2))], True, (0, 255, 255))

    approach = -1
    car = 0
    ped = 0

    for b in boxes:
        if b.label == 'car' and car != 1:
            centerx = (b.left + b.right)/2
            centery = b.bot
            pt = geometry.Point(centerx, centery)
            if poly.contains(pt):
                car = 1
        elif b.label == 'ped' and ped != 1:
            centerx = (b.left + b.right)/2
            centery = b.bot
            pt = geometry.Point(centerx, centery)
            if poly.contains(pt):
                ped = 1
        if car == 1 and ped == 1:
            break

    for box in boxes:
        if box.label == 'ped' and len(box.prediction) > 5:
            leftx = 0
            rightx = 0
            centerx = 0
            centery = 0
            for i in range(len(box.prediction)-3, len(box.prediction)):
                leftx += box.prediction[i].left
                rightx += box.prediction[i].right
                centerx += (box.prediction[i].left + box.prediction[i].right)/2
                centery += box.prediction[i].bot
            leftx /= 3
            rightx /= 3
            centerx /= 3
            centery /= 3
            curcenter = [(box.left + box.right)/2, (box.top + box.bot)/2]
            if abs(centerx - curcenter[0]) > 20 or abs(centery - curcenter[1]) > 20:
                if poly.contains(geometry.Point(leftx, centery)) or poly.contains(geometry.Point(rightx, centery)) or poly.contains(geometry.Point(curcenter[0], curcenter[1])):
                    approach = 1
                else:
                    approach = 0
                bayesian = intersection(car, ped, approach)
                #print('box: ' + str(box.id) + ', car: ' + str(car) + ', ped: ' + str(ped) + ', approach: ' + str(approach) + ', bayes: ' + str(bayesian))
                if (bayesian >= 0.85):
                    #print('box: ' + str(box.id) + ', car: ' + str(car) + ', ped: ' + str(ped) + ', approach: ' + str(approach) + ', bayes: ' + str(bayesian))
                    cv2.rectangle(frame, (box.left, box.top), (box.right, box.bot), (0, 0, 255), 2)
    
    return frame

def main():
    global boxes, objid, lostBoxes

    output = []

    cam = video.create_capture('cut1.mp4')
    content = readjson('video.json')

    framecount = 0
    for frames in content:
        for box in lostBoxes:
            if len(box.prediction) < 1:
               lostBoxes.remove(box)
            else: 
                oldpred = box.prediction.pop(0)
                #newBox = Box(oldpred.top, oldpred.left, oldpred.bot, oldpred.right, box.label);
                #box.history.append(newBox)
                if len(box.prediction) < 1:
                   lostBoxes.remove(box)
        newboxes = []
        ret, frame = cam.read()

        #frame = optflow(frame)

        for obj in frames:
            if obj['confidence'] >= 0.15 and obj['label'] in objlist:
                newbox = Box(obj['topleft']['y'], obj['topleft']['x'], obj['bottomright']['y'], 
                    obj['bottomright']['x'], obj['label']) 
                if framecount == 0:
                    newbox.id = objid
                    objid += 1
                newboxes.append(newbox)
        
        for box in newboxes:
            box.findFeatures()
            box.matchPoints()
            cv2.rectangle(frame, (box.left, box.top), (box.right, box.bot), box.color, 3)
            draw_str(frame, (box.left, box.top-10), str(box.id))
            if len(box.history) > 1:
                center1 = ((box.history[0].left+box.history[0].right)/2, (box.history[0].top+box.history[0].bot)/2)
                last = len(box.history) - 1
                center2 = ((box.history[last].left+box.history[last].right)/2, (box.history[last].top+box.history[last].bot)/2)
                cv2.arrowedLine(frame, center1, center2, (0, 0, 225), 2)
                box.prediction = []
                for i in range(0, last):
                    prediction_lefttop = (2*box.history[i].left - box.history[0].left, 2*box.history[i].top - box.history[0].top)
                    prediction_rightbot = (2*box.history[i].right - box.history[0].right, 2*box.history[i].bot - box.history[0].bot)
                    box.prediction.append(Box(prediction_lefttop[1], prediction_lefttop[0], prediction_rightbot[1], prediction_rightbot[0], box.label))
                    if (i%prediction_rate == 0) and (i > prediction_range_begin):
                        cv2.rectangle(frame, prediction_lefttop, prediction_rightbot, box.color, 1)

        lostBoxes = lostBoxes + boxes
        # for lostBox in lostBoxes:
        #     lostBox.history = []
        boxes = newboxes

        frame = calcBayesianNetwork(frame)

        jsonBoxes = {}
        for box in boxes:
            jsonBoxes[box.id] = {
                "label": box.label,
                "left": box.left,
                "top": box.top,
                "right": box.right,
                "bot": box.bot
            }
        output.append(jsonBoxes);
        # print output

        cv2.imshow('track', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        framecount += 1

    with open('tracking.json', 'w') as outfile:
        json.dump(output, outfile)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
