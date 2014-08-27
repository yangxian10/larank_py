#2014-8-25
#data driven tracking
__author__ = 'yangxian'

import numpy as np
import math
import larank
import pca_net

class tracker(object):
    def __init__(self):
        self.__svmBudgetSize = 100
        self.__svmC = 100.0
        self.__searchRadius = 30
        self.__learner = larank.SVM_model(self.__svmC, self.__svmBudgetSize)
        self.__V = []

    def init(self, inimg, box):
        rects = self.get_rects(inimg, box, self.__searchRadius, 1)
        train_imgs = []
        for i in range(len(rects)):
            x, y, w, h = rects[i]
            train_imgs.append(inimg[y:(y+h), x:(x+w)])
        f, self.__V, blk = pca_net.train(train_imgs)
        print 'finish train filter V'
        self.update_learner(inimg, box)
        print 'finish init'

    def process_frame(self, inimg, box):
        # detect
        rects = self.get_rects(inimg, box, self.__searchRadius, 0)
        features = self.get_feature(inimg, rects)
        scores = self.__learner.eval(features, rects)
        best_score = max(scores)
        best_index = scores.index(best_score)
        # learn
        self.update_learner(inimg, rects[best_index])
        return rects[best_index]

    def update_learner(self, img, box):
        rects = self.get_rects(img, box, self.__searchRadius, 1)
        features = self.get_feature(img, rects)
        self.__learner.update(features, rects, 0)
        return rects

    def get_rects(self, img, box, radius, train_flag):
        rects = []
        xb, yb, wb, hb = box
        if train_flag:
            rstep = float(2*radius)/5
            tstep = 2*math.pi/16
            rects.append(box)
            for ir in range(1,6):
                for it in range(16):
                    dx = round(ir*rstep*math.cos(it*tstep))
                    dy = round(ir*rstep*math.sin(it*tstep))
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
        out_rects = []
        for i in range(len(rects)):
            x, y, w, h = rects[i]
            if x>=0 and y>=0 and (x+w)<=img_w and (y+h)<=img_h:
                out_rects.append(rects[i])
        return out_rects

    def get_feature(self, img, rects):
        test_imgs = []
        for i in range(len(rects)):
            x, y, w, h = rects[i]
            test_imgs.append(img[y:(y+h), x:(x+w)])
        features = []
        for i in range(len(test_imgs)):
            f, blk = pca_net.feaExt([test_imgs[i]], self.__V)
            features.append(f)
        return features