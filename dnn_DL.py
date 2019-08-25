#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:11:40 2018
@author: wsw
"""

# construct simple DNN
import numpy as np
import matplotlib.pyplot as plt
import sys


def generate_data():
    x = np.linspace(-2, 2, 100)[np.newaxis, :]
    # numpy.random.normal(loc=0.0, scale=1.0, size=None)--给出均值为loc，标准差为scale的高斯随机数（场）
    noise = np.random.normal(0.0, 0.5, size=(1, 100))
    y = x ** 2 + noise
    return x, y


class DNN():
    def __init__(self, input_nodes=1, hidden1_nodes=4, hidden2_nodes=4, output_nodes=1):
        self.input_nodes = input_nodes
        self.hidden1_nodes = hidden1_nodes
        self.hidden2_nodes = hidden2_nodes
        self.output_nodes = output_nodes
        self.build_DNN()

    def build_DNN(self):
        np.random.seed(1)
        # Layer1 parameter
        self.w1 = np.random.normal(0.0, 0.1, size=(self.hidden1_nodes, self.input_nodes))
        self.b1 = np.zeros(shape=(self.hidden1_nodes, 1))
        # Layer2 parameter
        self.w2 = np.random.normal(0.0, 0.2, size=(self.hidden2_nodes, self.hidden1_nodes))
        self.b2 = np.ones(shape=(self.hidden2_nodes, 1))
        # Layer3 parameter
        self.w3 = np.random.normal(0.0, 0.5, size=(self.output_nodes, self.hidden2_nodes))
        self.b3 = np.zeros(shape=(self.output_nodes, 1))

    def forwardPropagation(self, inputs):
        self.z1 = np.matmul(self.w1, inputs) + self.b1
        self.a1 = 1 / (1 + np.exp(-self.z1))
        self.z2 = np.matmul(self.w2, self.a1) + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))
        self.z3 = np.matmul(self.w3, self.a2) + self.b3
        self.a3 = self.z3

    def backwardPropagation(self, da, a, a_1, w, b, last=False):
        '''
        da:current layer activation output partial devirate result
        a:current layer activation output
        a_1:previous layer of current layer activation output
        w:current parameter
        b:current bias
        '''
        # dz = da/dz
        if last:
            dz = da
        else:
            dz = a * (1 - a) * da
        da_1 = np.matmul(w.T, dz)
        # dw = dz/dw
        nums = da.shape[1]
        dw = np.matmul(dz, a_1.T) / nums
        db = np.mean(dz, axis=1, keepdims=True)
        # da_1 = dz/da_1


        w -= 0.5 * dw
        b -= 0.5 * db
        return da_1


    def train(self, x, y, max_iter=50000):
        for i in range(max_iter):
            # step01: 前向传播
            self.forwardPropagation(x)
            # step02: 损失函数
            kk = (self.a3 - y)**2
            loss = 0.5 * np.mean((self.a3 - y) ** 2)
            # 求第L层的梯度损失
            da = self.a3 - y
            # step03: 后向传播--更新参数
            da_2 = self.backwardPropagation(da, self.a3, self.a2, self.w3, self.b3, True)
            da_1 = self.backwardPropagation(da_2, self.a2, self.a1, self.w2, self.b2)
            da_0 = self.backwardPropagation(da_1, self.a1, x, self.w1, self.b1)
            self.view_bar(i + 1, max_iter, loss)
        dd = self.a3
        return self.a3

    def view_bar(self, step, total, loss):
        rate = step / total
        rate_num = int(rate * 40)
        r = '\rstep-%d loss value-%.4f[%s%s]\t%d%% %d/%d' % (step, loss, '>' * rate_num, '-' * (40 - rate_num),
                                                             int(rate * 100), step, total)
        # print(r)
        sys.stdout.write(r)
        sys.stdout.flush()


if __name__ == '__main__':
    x, y = generate_data()
    m = np.shape(x)
    n = np.shape(y)
    plt.scatter(x, y, c='r')
    # plt.ion()
    # print('plot')
    dnn = DNN()
    predict = dnn.train(x, y)
    plt.plot(x.flatten(), predict.flatten(), '-')
    # plt.plot(x, predict, '-')
    plt.show()