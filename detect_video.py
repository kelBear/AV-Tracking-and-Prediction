from darkflow.net.build import TFNet
import cv2
import os
from time import time as timer
import sys
import pyximport; pyximport.install()
import cy_yolo2_findboxes
import numpy as np
import json

def findboxes(tfnet, net_out):
    # meta
    meta = tfnet.meta
    boxes = list()
    boxes = cy_yolo2_findboxes.box_constructor(meta,net_out)
    return boxes

def process_bounding_boxes(tfnet, net_out, im, frame, save = True):
    """
    Takes net output, draw net_out, save to disk
    """
    boxes = findboxes(tfnet, net_out)

    # meta
    meta = tfnet.meta
    threshold = meta['thresh']
    colors = meta['colors']
    labels = meta['labels']
    if type(im) is not np.ndarray:
        imgcv = cv2.imread(im)
    else: imgcv = im
    h, w, _ = imgcv.shape
    
    resultsForJSON = []
    for b in boxes:
        boxResults = tfnet.framework.process_box(b, h, w, threshold)
        if boxResults is None:
            continue
        left, right, top, bot, mess, max_indx, confidence = boxResults
        thick = int((h + w) // 300)
        resultsForJSON.append({"frame: ": frame, "label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})

        cv2.rectangle(imgcv,
            (left, top), (right, bot),
            colors[max_indx], thick)
        cv2.putText(imgcv, mess, (left, top - 12),
            0, 1e-3 * h, colors[max_indx],thick//3)

    textJSON = json.dumps(resultsForJSON)
    with open("video.json", "a") as f:
        f.write(textJSON + "\n") 

    return imgcv

options = {"model": "cfg/yolo-voc.cfg", "load": "bin/yolo-voc.weights", "threshold": float(sys.argv[2]), "gpu": 1.0, "demo": sys.argv[1]}

tfnet = TFNet(options)
print(tfnet.FLAGS)

camera = cv2.VideoCapture(tfnet.FLAGS.demo)
_, frame = camera.read()
height, width, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = round(camera.get(cv2.CAP_PROP_FPS))
videoWriter = cv2.VideoWriter('video.mp4', fourcc, fps, (width, height))
buffer_inp = list()
buffer_pre = list()
elapsed = int()
start = timer()

while camera.isOpened():
	elapsed += 1
        _, frame = camera.read()
        if frame is None:
            print ('\nEnd of Video')
            break
        preprocessed = tfnet.framework.preprocess(frame)
        buffer_inp.append(frame)
        buffer_pre.append(preprocessed)
        
        # Only process and imshow when queue is full
        if elapsed % tfnet.FLAGS.queue == 0:
            feed_dict = {tfnet.inp: buffer_pre}
            net_out = tfnet.sess.run(tfnet.out, feed_dict)
            for img, single_out in zip(buffer_inp, net_out):
                #postprocessed = tfnet.framework.postprocess(single_out, img, False)
                postprocessed = process_bounding_boxes(tfnet, single_out, img, elapsed)
                videoWriter.write(postprocessed)
            # Clear Buffers
            buffer_inp = list()
            buffer_pre = list()

        if elapsed % 5 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('{0:3.3f} FPS'.format(
                elapsed / (timer() - start)))
            sys.stdout.flush()

sys.stdout.write('\n')
videoWriter.release()
camera.release()

