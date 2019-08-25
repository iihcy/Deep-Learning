# -*- coding: utf-8 -*-

"""
Created on Wed Nov 21 17:32:28 2018

@author: zhen
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# C:\Users\Administrator\PycharmProjects001-20180827\Deep learning\CNN


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main_model():
    mnist = input_data.read_data_sets('MNIST-data/', one_hot=True)
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 784])
    print(x)
    y = tf.placeholder(tf.float32, [None, 10])
    print(y)
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一层卷积核
    W_conv = weight_variable([5, 5, 1, 16])
    b_conv = bias_variable([16])
    h_conv = tf.nn.relu(conv2d(x_image, W_conv) + b_conv)
    h_pool = max_pool_2x2(h_conv)

    # 第二层卷积核
    W_conv2 = weight_variable([5, 5, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层
    W_fc = weight_variable([7 * 7 * 32, 512])
    b_fc = bias_variable([512])
    h_pool_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 32])
    h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc) + b_fc)

    # 防止过拟合，使用Dropout层
    keep_prob = tf.placeholder(tf.float32)
    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

    # Softmax分类
    W_fc2 = weight_variable([512, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc_drop, W_fc2) + b_fc2)

    # 定义损失函数
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 训练
    tf.global_variables_initializer().run()
    # 总共有100万样本，每批次50，执行20000轮
    for i in range(2000):
        # print('CNN_Itertive:', i)
        batch = mnist.train.next_batch(50)
        if (i % 20 == 0):
            train_accurary = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            print('The step %d training accuracy: %.4f' % (i, train_accurary))
        train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

    print('***********************')
    print("test accuracy %.4f" % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
    print('***********************')


if __name__ == '__main__':
    main_model()