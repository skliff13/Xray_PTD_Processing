#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import sys
import os
import glob
import pandas as pd
import numpy as np
import skimage.io as skio
import skimage.color as skcl
import caffe
import scipy
import scipy.misc
from skimage import transform



class CaffeBatchClassifier:
    gpuID = -1

    pathDirModel = None
    pathDirDB = None

    pathModelWeihts = None
    pathModelParams = None
    pathDBMeanProto = 'mean.binaryproto'

    prefMean = 'train_db/mean.binaryproto'
    prefModelP = 'deploy.prototxt'
    prefModelW = 'snapshot_iter_'

    sizBatch = None
    dataShape = None
    mu = None
    mean = None

    def __init__(self, gpuID=-1):
        self.net=None
        self.gpuID = gpuID

    def _checkFile(self, fnPath, isRaiseException=True):
        isOk = os.path.isfile(fnPath)
        if isRaiseException and (not isOk):
            raise Exception('ERROR: cant find file [%s]' % fnPath)
        return isOk

    def _checkDir(self, dirPath, isRaiseException=True):
        isOk = os.path.isdir(dirPath)
        if isRaiseException and (not isOk):
            raise Exception('ERROR: cant find diectory [%s]' % dirPath)
        return isOk

    def loadData(self):
        pass

    def loadMeanInfo(self, newShape=None):
        blobMean=caffe.proto.caffe_pb2.BlobProto()
        path_mean_proto = os.path.join(self.pathDirModel, self.pathDBMeanProto)
        with open(path_mean_proto,'rb') as f:
            blobMean.ParseFromString(f.read())
            arrMean=np.array(caffe.io.blobproto_to_array(blobMean))
            if newShape is not None:
                tarrMean = arrMean[0]
                if tarrMean.shape[0] > 1:
                    isRGB = True
                    tarrMean = tarrMean.transpose((1,2,0))
                else:
                    isRGB = False
                    tarrMean = tarrMean.reshape(tarrMean.shape[1:])

                tarrMean = scipy.misc.imresize(tarrMean.astype(np.uint8), newShape)
                if isRGB:
                    tarrMean = tarrMean.transpose((2,0,1))
                else:
                    tarrMean = tarrMean.reshape([1] + list(tarrMean.shape))
                self.mean = tarrMean
            else:
                self.mean = arrMean[0]
            self.mu = np.mean(arrMean[0],axis=(1,2))

    def loadModel(self, pathDirModel, modelWeigthsStateId=None):
        # (1) Set basic paths
        self.pathDirModel = pathDirModel
        self._checkDir(self.pathDirModel)

        # (2) Eval other paths
        self.pathModelParams = os.path.join(self.pathDirModel, self.prefModelP)

        # (2.1) try automatically select latest model from directory by max #IterationNumber
        if modelWeigthsStateId is not None:
            self.pathModelWeihts = os.path.join(self.pathDirModel, '%s.caffemodel' % modelWeigthsStateId)
        else:
            tlstModels = glob.glob('%s/*.caffemodel' % self.pathDirModel)

            if len(tlstModels) < 1:
                raise Exception('Cant find *.caffemodel files in directory [%s]' % self.pathDirModel)

            tlstIdx = np.array([int(os.path.basename(os.path.splitext(xx)[0]).split('_iter_')[1]) for xx in tlstModels])
            tidx = np.argsort(tlstIdx)[-1]
            self.pathModelWeihts = tlstModels[tidx]
            print(':: Found model [%s] in directory [%s]' % (os.path.basename(self.pathModelWeihts), self.pathDirModel))

        self._checkFile(self.pathModelParams)
        self._checkFile(self.pathModelWeihts)

        if self.gpuID<0:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(self.gpuID)
        self.net = caffe.Net(self.pathModelParams,
                             self.pathModelWeihts,
                             caffe.TEST)
        self.sizBatch = 1
        self.dataShape = self.net.blobs['data'].data.shape[1:]
        self.loadMeanInfo(newShape=self.dataShape)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', self.mu)
        if self.dataShape[0]>1:
            self.transformer.set_channel_swap('data', (2,1,0))

    def isInitialized(self):
        return self.net is not None

    def loadAndTransformImage(self, imagePath):
        self._checkFile(imagePath)
        timg = skio.imread(imagePath)
        if (timg.shape[0] != self.dataShape[1]) or (timg.shape[1] != self.dataShape[2]):
            timg = transform.resize(timg, (self.dataShape[1], self.dataShape[2]))
            timg[timg < 0] = 0
            timg[timg > 1] = 1
            timg = (timg * 255).astype(np.uint8)
        if len(timg.shape)<3:
            timg = timg.reshape(list(timg.shape) + [1])
        return self.transformer.preprocess('data', timg)


def calcSoftMax(parr, paxis=1):
    parr = parr - np.max(parr, axis=paxis, keepdims=True)
    tret = np.exp(parr) / np.sum(np.exp(parr), axis=paxis)[:, None]
    return tret


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: %s {/path/to/Directory_With_Model} {/path/to/Directory_With_Images}' % os.path.basename(sys.argv[0]))
        sys.exit(1)

    pathToDigitsJobDir_with_Model = sys.argv[1]
    pathDirWithImages = sys.argv[2]

    lstImages = glob.glob('%s/*.dcm.png' % pathDirWithImages)
    classifier = CaffeBatchClassifier(gpuID=-1)
    classifier.loadModel(pathDirModel=pathToDigitsJobDir_with_Model)

    for pathToImageFile in lstImages:
        timg = classifier.loadAndTransformImage(pathToImageFile)
        newShape = [classifier.sizBatch] + list(timg.shape)
        classifier.net.blobs['data'].reshape(*newShape)
        classifier.net.blobs['data'].data[0] = timg
        ret = classifier.net.forward()['score'][0]

        tshape = ret.shape
        ret = ret.reshape(tshape[0],-1)
        ret = calcSoftMax(ret.transpose((1,0)), paxis=1)
        ret = ret.transpose((1,0)).reshape(tshape)
        tpathOut = '%s-mask.png' % (pathToImageFile)
        skio.imsave(tpathOut, (255.*ret[1]).astype(np.uint8))
