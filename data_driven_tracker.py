#2014-8-25
#data driven tracking
__author__ = 'yangxian'

import numpy as np
import math
import larank

__svmBudgetSize = 100
__svmC = 100.0
__searchRadius = 30
__learner = larank.SVM_model(__svmC, __svmBudgetSize)

def init(inimg, box):
    update_learner(inimg, box)

def process_frame(inimg, box):
    global  __learner
    # detect
    rects = get_rects(inimg, box, __searchRadius, 0)
    features = get_feature(inimg, rects)
    scores = __learner.eval(features, rects)
    best_score = max(scores)
    best_index = scores.index(best_score)
    # learn
    update_learner(inimg, rects[best_index])
    return rects[best_index]

def update_learner(img, box):
    global __learner
    rects = get_rects(img, box, __searchRadius, 1)
    features = get_feature(img, rects)
    __learner.update(features, rects, 0)

def get_rects(img, box, radius, train_flag):
    rects = []
    xb,yb,wb,hb = box
    if train_flag:
        rstep = float(2*radius)/5
        tstep = 2*math.pi/16
        rects.append(box)
        for ir in range(1,6):
            for it in range(16):
                dx = ir*rstep*math.cos(it*tstep)
                dy = ir*rstep*math.sin(it*tstep)
                x = xb + dx
                y = yb + dy
                w = wb
                h = hb
                rects.append([x,y,w,h])
    else:
        r2 = radius ** 2
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if i**2+j**2 > r2:
                    continue
                x = xb + j
                y = yb + i
                w = wb
                h = hb
                rects.append([x,y,w,h])
    img_h, img_w = np.shape(img)
    for i in range(len(rects)):
        x,y,w,h = rects[i]
        if x<0 or y<0 or (x+w)>img_w or (y+h)>img_h:
            rects.pop(i)
    return rects

def get_feature(img, box):
    x,y,w,h = box
    feature = np.zeros((__treeNum,h,w))
    weights = 2**np.array(range(__fernLenth-1,-1,-1))
    for i in range(__treeNum):
        featuretemp = np.zeros((h,w))
        for j in range(__fernLenth):
            signimg = brief.getftr(imgPyr, box, brief_ftr[i,j])
            featuretemp += weights[j]*signimg
        feature[i] = featuretemp
    return feature