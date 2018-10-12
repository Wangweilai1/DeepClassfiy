# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


#-----------------构建网络----------------------
def build_network(width, height, channel, classNum):
    x = tf.placeholder(tf.float32,shape=[None,width,height,channel],name='input')
    y_ = tf.placeholder(tf.int32,shape=[None,],name='labels_placeholder')

    #第一个卷积层(100->50)
    conv1=tf.layers.conv2d(
          inputs=x,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    #第二个卷积层(50->25)
    conv2=tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    #第三个卷积层(25->12)
    conv3=tf.layers.conv2d(
          inputs=pool2,
          filters=128,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    #第四个卷积层(12->6)
    conv4=tf.layers.conv2d(
          inputs=pool3,
          filters=128,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

    #全连接层
    dense1 = tf.layers.dense(inputs=re1, 
                          units=1024, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2= tf.layers.dense(inputs=dense1, 
                          units=512, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense3= tf.layers.dense(inputs=dense2, 
                            units=classNum, 
                            activation=None,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

    finaloutput = tf.nn.softmax(dense3, name="softmax")
    prediction_labels = tf.argmax(finaloutput, axis=1, name="output")

    return dict(
        x=x,
        y=y_,
        finaloutput=finaloutput,
        prediction_labels=prediction_labels,
    )

#-----------------构建VGG16网络----------------------
def VGG16(width, height, channel, classNum):
    x = tf.placeholder(tf.float32,shape=[None,width,height,channel],name='input')
    y_ = tf.placeholder(tf.int32,shape=[None,],name='labels_placeholder')

    #conv1_1
    conv1_1=tf.layers.conv2d(
          inputs=x,
          filters=64,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    #conv1_2
    conv1_2=tf.layers.conv2d(
          inputs=conv1_1,
          filters=64,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1=tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)

    #conv2_1
    conv2_1=tf.layers.conv2d(
          inputs=pool1,
          filters=128,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    #conv2_2
    conv2_2=tf.layers.conv2d(
          inputs=conv2_1,
          filters=128,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2=tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)
    
    #conv3_1
    conv3_1=tf.layers.conv2d(
          inputs=pool2,
          filters=256,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    #conv3_2
    conv3_2=tf.layers.conv2d(
          inputs=conv3_1,
          filters=256,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))  
    #conv3_3
    conv3_3=tf.layers.conv2d(
          inputs=conv3_2,
          filters=256,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3=tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2, 2], strides=2)
    
    #conv4_1
    conv4_1=tf.layers.conv2d(
          inputs=pool3,
          filters=512,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    #conv4_2
    conv4_2=tf.layers.conv2d(
          inputs=conv4_1,
          filters=512,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    #conv4_3
    conv4_3=tf.layers.conv2d(
          inputs=conv4_2,
          filters=512,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool4=tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2, 2], strides=2)
    
    #conv5_1
    conv5_1=tf.layers.conv2d(
          inputs=pool4,
          filters=512,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    #conv5_2
    conv5_2=tf.layers.conv2d(
          inputs=conv5_1,
          filters=512,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    #conv5_3
    conv5_3=tf.layers.conv2d(
          inputs=conv5_2,
          filters=512,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool5=tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2, 2], strides=2)
    
    shape = int(np.prod(pool5.get_shape()[1:]))
    re1 = tf.reshape(pool5, [-1, shape])

    #FC
    dense1 = tf.layers.dense(inputs=re1, 
                          units=4096, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2= tf.layers.dense(inputs=dense1, 
                          units=4096, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense3= tf.layers.dense(inputs=dense2, 
                            units=classNum, 
                            activation=tf.nn.relu,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

    finaloutput = tf.nn.softmax(dense3, name="softmax")
    prediction_labels = tf.argmax(finaloutput, axis=1, name="output")

    return dict(
        x=x,
        y=y_,
        finaloutput=finaloutput,
        prediction_labels=prediction_labels,
    )