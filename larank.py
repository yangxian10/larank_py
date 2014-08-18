__author__ = 'yangxian'

import numpy as np
import cv2

class support_pattern(object):
    pass

class support_vector(object):
    pass

class svmmodel(object):
    def __init__(self, svmC=100.0, svmBudgetSize=100):
        self.__kTileSize = 30
        self.__kMaxSVs = 2000
        self.__C = svmC
        if svmBudgetSize > 0:
            N = svmBudgetSize + 2
        else:
            N = self.__kMaxSVs
        self.__K = np.mat(np.zeros((N,N)))
        self.__debugImage = np.zeros((800,600))

    def test(self):
        a = self.__K
        b = self.__C
        print b
        print "hello"
        cv2.namedWindow("debug")
        cv2.imshow("debug", self.__debugImage)
        cv2.waitKey(0)

    def loss(self, y1, y2):
        pass

    def evaluate(self, ):
        pass



