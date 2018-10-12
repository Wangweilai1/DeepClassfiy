# -*- coding: utf-8 -*-

from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
from model import build_network, VGG16
from tensorflow.python.framework import graph_util

path='./flower_photos/'
save_pb_file_path = "classfiy.pb"

#将所有的图片resize 100*100
w=100
h=100
c=3


#读取图片
def read_img(path):
    cate   = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs   = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the image: {0} -> label: {1}'.format(im, idx))
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
data,label=read_img(path)


#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]


#将所有数据分为训练集和验证集
ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]

#-----------------构建网络----------------------
graph = build_network(w, h, c, 5)
#graph = VGG16(w, h, c, 5)

#---------------------------计算Loss---------------------------

loss=tf.losses.sparse_softmax_cross_entropy(labels=graph['y'],logits=graph['finaloutput'])
train_op=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
#train_op=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correct_prediction = tf.equal(tf.cast(tf.argmax(graph['finaloutput'],1),tf.int32), graph['y'])    
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)

    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


#训练和验证数据

n_epoch=50
batch_size=2
sess=tf.InteractiveSession()  
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    print(" ----------------------------------- Number of Epoch: {0} ------------------------------------ ".format(epoch))
    start_time = time.time()
    
    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={graph['x']: x_train_a, graph['y']: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
    print("   train loss: %f" % (train_loss/ n_batch))
    print("   train acc: %f" % (train_acc/ n_batch))
    
    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={graph['x']: x_val_a, graph['y']: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
    print("   validation loss: %f" % (val_loss/ n_batch))
    print("   validation acc: %f" % (val_acc/ n_batch))

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
with tf.gfile.FastGFile(save_pb_file_path, mode='wb') as f:
    f.write(constant_graph.SerializeToString())

sess.close()
