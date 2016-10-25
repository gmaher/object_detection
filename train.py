import numpy as np
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Flatten
from keras.objectives import mean_absolute_error
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from util import *

X = np.random.randn(1,400,400,3)
y = np.asarray([
    [100.5,50.6,50,70],
    [250,150,90,20]
])

Nbatch,H,W,C = X.shape
Nfilter = 16
Wfilter = 3
lr = 1e-4
Niter = 100

sess = tf.Session()
K.set_session(sess)

img = tf.placeholder(tf.float32, shape=(None,H,W,C))
true_box = tf.placeholder(tf.float32, shape=(None,H/2,W/2,4))
true_conf = tf.placeholder(tf.float32, shape=(None,H/2,W/2,1))

out = MaxPooling2D(pool_size=(2,2))(img)

out = Convolution2D(Nfilter, Wfilter, Wfilter,
    activation='relu', border_mode='same')(out)

out = Convolution2D(Nfilter, Wfilter, Wfilter,
    activation='relu', border_mode='same')(out)

out = MaxPooling2D(pool_size=(2,2))(img)

out = Convolution2D(Nfilter, Wfilter, Wfilter,
    activation='relu', border_mode='same')(out)

out_box = Convolution2D(4, Wfilter, Wfilter,
    activation='relu', border_mode='same')(out)

out_conf = Convolution2D(1, Wfilter, Wfilter,
    activation='sigmoid', border_mode='same')(out)

loss_vec = tf.square(true_box-out_box)
loss = tf.reduce_sum(tf.mul(true_conf,tf.reduce_sum(loss_vec,keep_dims=True)))\
 + tf.reduce_sum(tf.square(true_conf-out_conf))

opt = tf.train.AdamOptimizer(lr)
train = opt.minimize(loss)

yhat = sess.run(out_box,feed_dict={img:X})
S = yhat.shape[1]

P = np.zeros((1,S,S,1))
Y = np.zeros(yhat.shape)

###########
# Training
###########
init = tf.initialize_all_variables()
sess.run(init)

for step in xrange(Niter):
    P[:]=0.0
    Y[:]=0.0
    yhat = sess.run(out_box,feed_dict={img:X})
    boxes, inds = getBboxLabels(yhat,y,W,H,S)
    for bbox,best_index in zip(boxes,inds):
        P[0,best_index[0],best_index[1],0] = 1.0
        Y[0,best_index[0],best_index[1],:] = normalize_bbox(bbox,W,H,S)
        print bbox, Y[0,best_index[0],best_index[1],:], \
            yhat[0,best_index[0],best_index[1],:],best_index
    sess.run(train, feed_dict={img:X, true_box:Y, true_conf:P})

    if step%20 == 0:
        l = sess.run(loss, feed_dict={img:X, true_box:Y, true_conf:P})
        print "loss={}".format(l)
