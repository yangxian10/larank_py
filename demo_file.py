#2014-8-27
#data driven tracking
#main run by sequence
__author__ = 'yangxian'

import cv2
import numpy as np
import time
import data_driven_tracker as ddt

print 'python implemented by yang xian'
print 'Version 1.0'
print 'email yang_xian521@163.com'
print 'date 2014/8/25'

__sequence = 'girl'

cap = cv2.VideoCapture()
cap.open(0)
cv2.namedWindow('ddt')

cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240)

frames_file_path = 'sequences//' + __sequence + "//" + __sequence + "_frames.txt"
for line in open(frames_file_path):
    start_frame, end_frame = line.strip().split(',')
start_frame = int(start_frame)
end_frame = int(end_frame)
img_path = 'sequences//' + __sequence + "//imgs//img" + '%05d' % start_frame + '.png'
frame = cv2.imread(img_path)
grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow('ddt', frame)
ground_truth_path = 'sequences//' + __sequence + "//" + __sequence + "_gt.txt"
xgt, ygt, wgt, hgt = open(ground_truth_path).readline().strip().split(',')
box = [int(xgt), int(ygt), int(wgt), int(hgt)]

# initialization
print 'initialization tracker'

print 'init tracking box =x:',xgt,'y:',ygt,'w:',wgt,'h:',hgt
ddt_tracker = ddt.tracker()
ddt_tracker.init(grayframe, box)

# run
for i in range(start_frame, end_frame+1):
    img_path = 'sequences//' + __sequence + "//imgs//img" + '%05d' % i + '.png'
    frame = cv2.imread(img_path)
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    start = time.time()
#   grayframe = cv2.GaussianBlur(grayframe, (7,7), 1.5)
    rect = ddt_tracker.process_frame(grayframe, box)
    bx, by, bw, bh = rect
    #cv2.waitKey(50)
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh), (0,0,255))
    end = time.time()
    fps = 1.0/(end-start)
    fps_str = 'FPS:'+str(fps)
    cv2.putText(frame, fps_str, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))
    cv2.imshow('ddt', frame)
    print 'current tracking box =x:',bx,'y:',by,'w:',bw,'h:',bh
    key = cv2.waitKey(5)
    if key == 27:
        break
    elif key == ord('p'):
        cv2.waitKey(0)

cv2.destroyWindow('ddt')