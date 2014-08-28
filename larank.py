__author__ = 'yangxian'

import numpy as np
import sys
import cv2
import random

class Support_pattern(object):
    def __init__(self):
        self.x = []
        self.yv = []
        #self.images = []
        self.y = 0
        self.refCount = 0

class Support_vector(object):
    def __init__(self):
        self.x = []
        self.y = 0
        self.b = 0.0
        self.g = 0.0
        #self.image = []

class SVM_model(object):
    def __init__(self, svmC=100.0, svmBudgetSize=100):
        self.__kTileSize = 30
        self.__kMaxSVs = 2000
        self.__C = svmC
        self.__svmBudgetSize = svmBudgetSize
        if svmBudgetSize > 0:
            N = svmBudgetSize + 2
        else:
            N = self.__kMaxSVs
        self.__K = np.mat(np.zeros((N,N)))
        #self.__debugImage = np.zeros((800,600))
        self.__sps = []
        self.__svs = []
        #self.__debugModel = True

    '''
    def test(self):
        a = self.__K
        b = self.__C
        print b
        print "hello"
        #cv2.namedWindow("debug")
        #cv2.imshow("debug", self.__debugImage)
        self.debug()
        cv2.waitKey(0)
    '''

    def eval(self, features, rects):
        results = []
        for i in range(len(features)):
            res = self.evaluate(features[i], rects[i])
            results.append(res)
        return results

    def update(self, features, rects, centerindex):
        sp = Support_pattern()
        for i in range(len(rects)):
            sp.yv.append(rects[i])
        sp.x = features
        sp.y = centerindex
        sp.refCount = 0
        self.__sps.append(sp)

        self.process_new(len(self.__sps)-1)
        self.budget_maintenance()

        for i in range(10):
            self.reprocess()
            self.budget_maintenance()

    def loss(self, r1, r2):
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        xp1 = max(x1,x2)
        yp1 = max(y1,y2)
        xp2 = min(x1+w1, x2+w2)
        yp2 = min(y1+h1, y2+h2)
        if xp1 >= xp2 or yp1 >= yp2:
            return 0.0
        overlap = (xp2-xp1) * (yp2-yp1)
        score = float(overlap) / (w1*h1 + w2*h2 - overlap)
        return 1 - score

    def evaluate(self, x, yRect):
        f = 0.0
        for i in range(len(self.__svs)):
            f += self.__svs[i].b * (x.dot(self.__svs[i].x.x[self.__svs[i].y]))
        return f

    def budget_maintenance(self):
        if self.__svmBudgetSize > 0:
            while len(self.__svs) > self.__svmBudgetSize:
                self.budget_maintenance_remove()

    def reprocess(self):
        self.process_old()
        for i in range(10):
            self.optimize()

    '''
    def compute_dual(self):
        d = 0.0
        for i in range(len(self.__svs)):
            sv = self.__svs[i]
            d -= sv.b * self.loss(sv.x.yv[sv.y], sv.x.yv[sv.x.y])
            for j in range(len(self.__svs)):
                d -= 0.5 * sv.b * self.__svs[j].b * self.__K[i,j]
        return d
    '''

    def SMOstep(self, ipos, ineg):
        if ipos == ineg:
            return
        svp = self.__svs[ipos]
        svn = self.__svs[ineg]
        assert svp.x == svn.x
        sp = svp.x

        if svp.g - svn.g > 1e-5:
            kii = self.__K[ipos,ipos] + self.__K[ineg,ineg] - self.__K[ipos,ineg]
            ru = (svp.g - svn.g) / kii
            r = min(ru, self.__C*int(svp.y == sp.y)-svp.b)
            svp.b += r
            svn.b -= r
            for i in range(len(self.__svs)):
                svi = self.__svs[i]
                svi.g -= r * (self.__K[i,ipos] - self.__K[i,ineg])

        if abs(svp.b) < 1e-8:
            self.remove_support_vector(ipos)
            if ineg == len(self.__svs):
                ineg = ipos

        if abs(svn.b) < 1e-8:
            self.remove_support_vector(ineg)

    def min_gradient(self, index):
        sp = self.__sps[index]
        min_grad_index = -1
        min_grad_val = sys.float_info.max
        for i in range(len(sp.yv)):
            grad = -self.loss(sp.yv[i], sp.yv[sp.y]) - self.evaluate(sp.x[i], sp.yv[i])
            if grad < min_grad_val:
                min_grad_index = i
                min_grad_val = grad
        return min_grad_index, min_grad_val

    def process_new(self, index):
        sp = self.__sps[index]
        idp = self.add_support_vector(sp, sp.y, -self.evaluate(sp.x[sp.y], sp.yv[sp.y]))
        min_grad_index, min_grad_val = self.min_gradient(index)
        idn = self.add_support_vector(sp, min_grad_index, min_grad_val)
        self.SMOstep(idp, idn)

    def process_old(self):
        if len(self.__sps) == 0:
            return
        index = random.randint(0, len(self.__sps)-1)
        idp = -1
        maxGrad = -sys.float_info.max
        for i in range(len(self.__svs)):
            if self.__svs[i].x != self.__sps[index]:
                continue
            svi = self.__svs[i]
            if svi.g > maxGrad and svi.b < self.__C*int(svi.y == self.__sps[index].y):
                idp = i
                maxGrad = svi.g
        assert idp != -1
        if idp == -1:
            return

        min_grad_index, min_grad_val = self.min_gradient(index)
        idn = -1
        for i in range(len(self.__svs)):
            if self.__svs[i].x != self.__sps[index]:
                continue
            if self.__svs[i].y == min_grad_index:
                idn = i
                break
        if idn == -1:
            idn = self.add_support_vector(self.__sps[index], min_grad_index, min_grad_val)
        self.SMOstep(idp, idn)

    def optimize(self):
        if len(self.__sps) == 0:
            return
        index = random.randint(0, len(self.__sps)-1)
        idp = -1
        idn = -1
        maxGrad = -sys.float_info.max
        minGrad = sys.float_info.max
        for i in range(len(self.__svs)):
            if self.__svs[i].x != self.__sps[index]:
                continue
            svi = self.__svs[i]
            if svi.g > maxGrad and svi.b < self.__C*int(svi.y == self.__sps[index].y):
                idp = i
                maxGrad = svi.g
            if svi.g < minGrad:
                idn = i
                minGrad = svi.g
        assert idp != -1 and idn != -1
        if idp == -1 or idn == -1:
            print "!!!!!!!!!!"
            return
        self.SMOstep(idp, idn)

    def add_support_vector(self, x, y, g):
        sv = Support_vector()
        sv.b = 0.0
        sv.x = x
        sv.y = y
        sv.g = g
        index = len(self.__svs)
        self.__svs.append(sv)
        x.refCount += 1

        for i in range(index):
            self.__K[i,index] = self.__svs[i].x.x[self.__svs[i].y].dot(x.x[y])
            self.__K[index,i] = self.__K[i,index]
        self.__K[index,index] = x.x[y].dot(x.x[y])
        return index

    def swap_support_vectors(self, index1, index2):
        tmp = self.__svs[index1]
        self.__svs[index1] = self.__svs[index2]
        self.__svs[index2] = tmp

        row = self.__K[index1]
        self.__K[index1] = self.__K[index2]
        self.__K[index2] = row

        col = self.__K[:,index1]
        self.__K[:,index1] = self.__K[:,index2]
        self.__K[:,index2] = col

    def remove_support_vector(self, index):
        self.__svs[index].x.refCount -= 1
        if self.__svs[index].x.refCount == 0:
            for i in range(len(self.__sps)):
                if self.__sps[i] == self.__svs[index].x:
                    self.__sps.pop(i)
                    break
        if index < len(self.__svs)-1:
            self.swap_support_vectors(index, len(self.__svs)-1)
            index = len(self.__svs)-1
        self.__svs.pop()

    def budget_maintenance_remove(self):
        minVal = sys.float_info.max
        idn = -1
        idp = -1
        for i in range(len(self.__svs)):
            if self.__svs[i].b < 0.0:
                j = -1
                for k in range(len(self.__svs)):
                    if self.__svs[k].b >0.0 and self.__svs[k].x == self.__svs[i].x:
                        j=k
                        break
                val = self.__svs[i].b**2*(self.__K[i,i]+self.__K[j,j]-2*self.__K[i,j])
                if val < minVal:
                    minVal = val
                    idn = i
                    idp = j

        self.__svs[idp].b += self.__svs[idn].b
        self.remove_support_vector(idn)
        if idp == len(self.__svs):
            idp = idn
        if self.__svs[idp].b < 1e-8:
            self.remove_support_vector(idp)

        for i in range(len(self.__svs)):
            svi = self.__svs[i]
            svi.g = -self.loss(svi.x.yv[svi.y], svi.x.yv[svi.x.y]) - self.evaluate(svi.x.x[svi.y], svi.x.yv[svi.y])

    def debug(self):
        print "%s//%s--support patterns/vectors" % (len(self.__sps), len(self.__svs))
        self.update_debug_image()
        b = []
        for i in range(self.__svs):
            b.append(self.__svs[i].b)
        print "support vectors B:"
        print b
        print "------------------"

    def update_debug_image(self):
        #debug_image = np.zeros((800, 600))
        svs_size = len(self.__svs)
        max_val = np.max(self.__K)
        min_val = np.min(self.__K)
        K = np.array(self.__K.shape)
        for i in range(svs_size):
            for j in range(svs_size):
                K[i,j] = float(self.__K[i,j] - min_val) / (max_val - min_val)
        cv2.imshow("learner_K", K)

