#2014-8-25
#data driven tracking
#main run by camera
__author__ = 'yangxian'

import cv2
import numpy as np
import time
import data_driven_tracker as ddt


print 'python implemented by yang xian'
print 'Version 1.0'
print 'email yang_xian521@163.com'
print 'date 2014/8/125'

gotBB = False
drawing = False
bx = 0
by = 0
bw = 0
bh = 0


def mousehandle(event, x, y, flags, param):
    global drawing
    global gotBB
    global bx
    global by
    global bw
    global bh
    if event == cv2.cv.CV_EVENT_LBUTTONDOWN:
        drawing = True
        bx = x
        by = y
    elif event == cv2.cv.CV_EVENT_MOUSEMOVE:
        if drawing:
            bw = x-bx
            bh = y-by
    elif event == cv2.cv.CV_EVENT_LBUTTONUP:
        drawing = False
        if bw<0:
            bx += bw
            bw *= -1
        elif bh<0:
            by += bh
            bh *= -1
        gotBB = True
    return

cap = cv2.VideoCapture()
cap.open(0)
cv2.namedWindow('ddt')
cv2.setMouseCallback('ddt', mousehandle)

cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240)

# initialization
print 'initialization tracker'

flag = True

while (not gotBB):
    flag,frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh), (0,0,255))
    cv2.imshow('ddt', frame)
    key = cv2.waitKey(20)
    if key == 27:
        break

print 'init tracking box =x:',bx,'y:',by,'w:',bw,'h:',bh
box = [bx,by,bw,bh]
ddt_tracker = ddt.tracker()
ddt_tracker.init(grayframe, box)

# run
while (flag):
    flag, frame = cap.read()
    start = time.time()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayframe = cv2.GaussianBlur(grayframe, (7,7), 1.5)
    rect = ddt_tracker.process_frame(grayframe, box)
    bx,by,bw,bh
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

cv2.destroyWindow('dsr')
