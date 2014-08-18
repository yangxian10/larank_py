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
        self.__sps = []
        self.__svs = []

    def test(self):
        a = self.__K
        b = self.__C
        print b
        print "hello"
        #cv2.namedWindow("debug")
        #cv2.imshow("debug", self.__debugImage)
        self.debug()
        cv2.waitKey(0)

    def eval(self, ):
        pass

    def update(self, ):
        pass

    def loss(self, y1, y2):
        pass

    def evaluate(self, ):
        pass

    def budget_maintenance(self):
        pass

    def reprocess(self):
        pass

    def compute_dual(self):
        pass

    def SMOstep(self):
        pass

    def min_gradient(self):
        pass

    def process_new(self):
        pass

    def process_old(self):
        pass

    def optimize(self):
        pass

    def add_support_vector(self):
        pass

    def swap_support_vectors(self):
        pass

    def remove_support_vector(self):
        pass

    def budget_maintenance_remove(self):
        pass

    def debug(self):
        print "%s//%s--support patterns/vectors" % (len(self.__sps), len(self.__svs))
        self.update_debug_image()
        cv2.imshow("learner", self.__debugImage)

    def update_debug_image(self):
        pass

