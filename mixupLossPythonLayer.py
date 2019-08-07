#-*- coding:utf-8 -*-
#create by Fancy265
import numpy as np
import caffe

class LossDataLayer(caffe.Layer):
    def setup(self,bottom,top):
        params = eval(self.param_str)
        self.batch_size = params['batch_size']
        self.numclass = params['numclass']
        # check for all inputs
        if len(bottom) != 2:
            raise Exception("Need two inputs (scores and labels) to compute sigmoid crossentropy loss.")

    def forward(self,bottom,top):
        score=bottom[0].data
        y_a = bottom[1].data[:,0]
        y_b = bottom[1].data[:,1]
        lam = bottom[1].data[:,2]

        score = self.getSoftMaxScore(score)

        y_a = y_a.reshape(self.batch_size,1)
        y_b = y_b.reshape(self.batch_size, 1)

        y_a = self.to_onehot(y_a,self.numclass)
        y_b = self.to_onehot(y_b, self.numclass)

        lossvalue = self.get_loss(score, y_a, y_b, lam)

        top[0].data[...]=lossvalue

        lam = lam.reshape(self.batch_size,1)

        self.diff=(score-y_a)* lam + (score-y_b)*(1-lam)


    def reshape(self,bottom,top):
        # check input dimensions match between the scores and labels
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1,)


    def backward(self,top,propagate_down,bottom):
        bottom[0].diff[...] = self.diff

    def caculateLoss(self,score,label):
        labelonehot = label.reshape(self.batch_size, self.numclass)
        loss = np.sum(- labelonehot * np.log(score))

        return loss

    def get_loss(self, score, y_a, y_b, lam):

        ret = lam * self.caculateLoss(score, y_a) + (1 - lam) * self.caculateLoss(score, y_b)
        return np.mean(ret)

    def to_onehot(self, y, num_classes):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()

        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]

        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), y] = 1

        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)

        return categorical

    def get_label(self,y_a, y_b, lam):
        label = y_a * lam + y_b * (1 - lam)
        return label

    def getSoftMaxScore(self,score):
        score = score.reshape(self.batch_size, self.numclass)
        eee = np.exp(score)
        fenmu = np.sum(eee, axis=1).reshape(self.batch_size, 1)
        score = np.true_divide(eee, fenmu)
        return score



