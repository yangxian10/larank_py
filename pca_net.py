import PCANet
#from sklearn import linear_model, decomposition, datasets
from numpy import *
import numpy
import copy

def filterbank(input_images, patch_size, num_filters):
    image_num = len(input_images)
    rx = zeros((patch_size**2, patch_size**2))
    for i in range(image_num):
        im = im2col(input_images[i], [patch_size, patch_size])
        meanvals = mean(im, axis=0)
        meanRemoved = im - meanvals
        covMat = cov(meanRemoved, rowvar=0)
        rx = rx + covMat
    rx = rx / (image_num*shape(meanRemoved)[1])
    eigVals, eigVects = linalg.eig(mat(rx))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(num_filters+1):-1]
    eigVectsPri = eigVects[:, eigValInd]
    return eigVectsPri

def pcaoutput(input_images, inimg_idx, patch_size, num_filters, eig_vects):
    image_num = len(input_images)
    mag = int((patch_size-1)/2)
    outImg = []
    for i in range(image_num):
        rows,cols = shape(input_images[i])
        img = zeros((rows+patch_size-1, cols+patch_size-1))
        img[(mag):(-1-mag+1), (mag):(-1-mag+1)] = input_images[i]
        im = im2col(img, [patch_size, patch_size])
        meanvals = mean(im, axis=0)
        im -= meanvals
        for j in range(num_filters):
            outImg.append((eig_vects[:,j].T*im.T).reshape(rows,cols))
    outimg_idx = kron(inimg_idx, ones((1,num_filters))[0])
    return outImg, outimg_idx

def hashinghist(img_idx, out_img):
    num_images = int(numpy.max(img_idx))
    map_weights = 2**array(range(PCANet.num_filters[-1]-1,-1,-1))
    patch_step = (1-PCANet.blk_overlap_ratio)*array(PCANet.hist_blocksize)
    patch_step = [int(round(n,0)) for n in patch_step]
    #print  "--", patch_step
    f = []
    bins = []
    numImgIn0 = []
    histsize = 2**PCANet.num_filters[-1]
    for idx in range(num_images+1):
        idx_span, = where(img_idx == idx)
        #bhist = []
        numImgIn0 = len(idx_span)/PCANet.num_filters[-1]
        #print '---numImgIn0 = ', numImgIn0
        for i in range(numImgIn0):
            T = zeros(shape(out_img[idx_span[0]]))
            for j in range(PCANet.num_filters[-1]):
                signmap = sign(out_img[idx_span[PCANet.num_filters[-1]*i+j]])
                signmap[signmap<=0]=0
                T += map_weights[j]*signmap
            #print 'T shape==', shape(T)
            TT = im2col(T,PCANet.hist_blocksize,patch_step)
            #bhisttemp = zeros((TT.shape[1], 2**PCANet.num_filters[-1]))
            bins = TT.shape[0]
            #print 'bins:',bins
            #print 'TT size:', shape(TT)
            for k in range(bins):
                bhisttemp,binstemp = histogram(TT[k,:],range(histsize+1))
                #print 'hist:', bhisttemp
                #f.append(bhisttemp*(histsize/sum(bhisttemp)))
                f.append(bhisttemp)
        #f.append(bhist)
    f=array(f).reshape(1,bins*numImgIn0*histsize)[0]
    blkIdx = kron(ones((1,numImgIn0))[0],kron(array(range(bins)),ones((1,histsize))[0]))
    return f,blkIdx
            
def im2col(in_img, patch_size, patch_step=[1,1]):
    rows, cols = shape(in_img)
    rowsize = len(range(0, rows-patch_size[0]+1, patch_step[0]))
    colsize = len(range(0, cols-patch_size[1]+1, patch_step[1]))
    length = rowsize*colsize
    out_img = mat(zeros((length,(patch_size[0]*patch_size[1]))))
    #print '---', shape(out_img)
    #step = len(range(0, cols-patch_size[1]+1, patch_step[1]))
    index = 0
    for i in range(0, rows-patch_size[0]+1, patch_step[0]):
        for j in range(0, cols-patch_size[1]+1, patch_step[1]):
            im = in_img[i:(i+patch_size[0]), j:(j+patch_size[1])]
            im = im.reshape(1,(patch_size[0]*patch_size[1]))
            out_img[index,:] = im
            index += 1
    return out_img  

def train(input_imgs):
    V = []
    numImg = len(input_imgs)
    imgIdx = range(numImg)
    print 'train image number:', numImg
    outimg = copy.copy(input_imgs)
    for stage in range(PCANet.num_stages):
        print 'pca training layer', stage 
        V.append(filterbank(input_imgs, PCANet.patch_size, PCANet.num_filters[stage]))
        if stage != PCANet.num_stages-1:
            outimg, imgIdx = pcaoutput(outimg, imgIdx, PCANet.patch_size, PCANet.num_filters[stage], V[stage])
    f = []
    #print 'outimg number', shape(outimg)
    #print 'outImgIdx', shape(imgIdx)
    print 'pca training hashing'
    for idx in range(numImg):
        outimgindex, = where(array(imgIdx) == idx)
        #print 'index:', outimgindex
        outimgtemp = []
        for i in outimgindex:
            outimgtemp.append(outimg[i])
        outimg_i, imgIdx_i = pcaoutput(outimgtemp,ones((1,len(outimgindex)))[0],PCANet.patch_size, PCANet.num_filters[-1], V[-1])
        ftemp, blkIdx = hashinghist(imgIdx_i, outimg_i)
        #print 'hist', ftemp
        f.append(ftemp)
    return f, V, blkIdx

def feaExt(input_img, V):
    outimg = copy.copy(input_img)
    #numImg = len(input_img)
    numImg = 1
    imgIdx = range(numImg)
    #print 'feature ext imgIdx:', imgIdx
    for stage in range(PCANet.num_stages):
        outimg, imgIdx = pcaoutput(outimg, imgIdx, PCANet.patch_size, PCANet.num_filters[stage], V[stage])
    #print 'outimg size:', shape(outimg)
    #print 'imgIdx:', shape(imgIdx)
    f, blkIdx = hashinghist(imgIdx, outimg)
    return f, blkIdx
