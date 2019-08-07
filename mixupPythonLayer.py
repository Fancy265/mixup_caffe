#-*- coding:utf-8 -*-
#create by Fancy265
import cv2
import numpy as np
import random
# import numbers
# from PIL import Image,ImageFilter,ImageEnhance

import caffe

new_width = 64
new_height = 64
isshuffle = True
# meanvalue = [123.68,116.779,103.939]
# scale = 0.0078125
class DataLayer(caffe.Layer):
    def setup(self,bottom,top):
        print("DataLayer  setup!!")
        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)
        # store input as class variables
        self.batch_size = params['batch_size']
        self.source_dir = params['source_dir']
        self.batch_loader = BatchLoader(params,None)
        top[0].reshape(self.batch_size,3,new_height,new_width)
        top[1].reshape(self.batch_size,3)

    def forward(self,bottom,top):
        """Get blobs and copy them into this layer's top blob vector."""
        # imgmaps = self.batch_loader.mixup_gen()
        # print("DataLayer  forward!!")
        trainX, trainY = self.batch_loader.batch_imgs()
        # print("trainX:",trainX.shape)
        # print("trainY:",trainY.shape)
        # print("trainY:", trainY)
        # print("top[0].data.shape:",top[0].data.shape)
        # print("top[1].data.shape:", top[1].data.shape)
        top[0].data[:, ...] = trainX
        top[1].data[:, ...] = trainY
        # print("DataLayer  forward!!")

    def reshape(self,bottom,top):
        pass
    def backward(self,top,propagate_down,bottom):
        pass


class BatchLoader(object):
    def __init__(self,params,result):
        self.result = result
        self.source = params['source_dir']
        self.batch_size = params['batch_size']
        self.new_width = new_width
        self.new_height = new_height
        self.alpha = 0.2

        self.isshuffle = isshuffle
        self.imagelist = open(self.source,'r').read().splitlines()
        self.sample_num = len(self.imagelist)
        if self.isshuffle:
            random.shuffle(self.imagelist)
        self._curIter = 0  # current iter
        self._totalIter = int(self.sample_num // (self.batch_size))


    def batch_imgs(self):

        if self._curIter >= self._totalIter:
            self._curIter = 0

        begin = self._curIter * self.batch_size
        end = (self._curIter+2)*self.batch_size
        if end<self.sample_num:
            curbatch = self.imagelist[begin:end]
        else:
            curbatch = self.imagelist[begin:self.sample_num]+self.imagelist[0:end-self.sample_num]

        targetBatch = curbatch[:self.batch_size]
        sourceBatch = curbatch[self.batch_size:]

        trainX, trainY = self.getData(targetBatch)
        trainX_, trainY_ = self.getData(sourceBatch)


        newX, newY = self.__data_generation(trainX, trainY, trainX_, trainY_)
        newX = np.transpose(newX,(0,3,1,2))
        self._curIter+=1

        return newX, newY

    def getData(self,batch):
        imglist = []
        labellist = []
        for i in range(len(batch)):

            txtmsg = batch[i].split(" ")
            if len(txtmsg)!=2:
                print("this input msg is: ",batch[i])
                continue

            filename = txtmsg[0]
            label = txtmsg[1]

            img = cv2.imread(filename)
            img = cv2.resize(img, (self.new_width, self.new_height))
            imglist.append(img)
            labellist.append(label)

        X = np.asarray(imglist,dtype=int)
        Y = np.asarray(labellist,dtype=int)
        return X, Y


    def __data_generation(self, X_train, Y_train, X_train1, Y_train1):
        if len(X_train)!=len(X_train1):
            print("please check X_train size and X_train1 size!!!!")
        if len(Y_train) != len(Y_train1):
            print("please check Y_train size and Y_train1 size!!!!")
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)
        Y_train = Y_train.reshape(self.batch_size, 1)
        Y_train1 = Y_train1.reshape(self.batch_size, 1)

        X1 = X_train
        X2 = X_train1
        X = X1 * X_l + X2 * (1 - X_l)

        y1 = Y_train
        y2 = Y_train1

        y = np.hstack((y1,y2,y_l))

        return X, y

