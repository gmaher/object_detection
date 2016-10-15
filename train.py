import numpy as np
from keras.layers import Input, Convolution2D, MaxPooling2D
from keras.objectives import mean_absolute_error
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

EPS = 1e-5

def calc_iou(box_pred,box_true):
    '''
    function to calculate intersection over union of two bounding boxes

    args:
        @a box_pred, box_true [x,y,w,h]
    '''
    Apred = box_pred[2]*box_pred[3]
    Atrue = box_true[2]*box_true[3]

    xr = min(box_pred[0]+box_pred[2]/2, box_true[0]+box_true[2]/2)
    xl = max(box_pred[0]-box_pred[2]/2, box_true[0]-box_true[2]/2)
    yb = min(box_pred[1]+box_pred[3]/2, box_true[1]+box_true[3]/2)
    yt = max(box_pred[1]-box_pred[3]/2, box_true[1]-box_true[3]/2)
    I = (xr-xl)*(yb-yt)
    U = Apred+Atrue-I

    return float(I)/(U)

def denormalize_bbox(bbox, W, H, S, i,j):
    '''
    Function to convert a bounding box from cell local coordinates to global image
    coordinates

    args:
        @a bbox: normalized bounding box coordinates [x,y,w,h]
            0 <= x,y,w,h <= 1
            where the origin of the cell is the top left corner
            and the bottom right corner is (1,1)
        @a W,H: image width, height in pixels
        @a S: output tensor height/width in pixels
        @a i,j: index of bbox in the output tensor
    '''
    x = float(W)/S*(j+bbox[0])
    y = float(H)/S*(i+bbox[1])
    w = bbox[2]*float(W)/S
    h = bbox[3]*float(H)/S
    return np.asarray([x,y,w,h])

def normalize_bbox(bbox,W,H,S):
    '''
    Function to convert a bounding box from global image coordinates to local
    cell coordinates

    args:
        @a bbox: bounding box coordinates [x,y,w,h] in pixels

        @a W,H: image width, height in pixels
        @a S: output tensor height/width in pixels
    '''
    scalex = float(S)/W
    scaley = float(S)/H
    x = scalex*bbox[0]
    x = x-int(x)
    y = scaley*bbox[1]
    y = y-int(y)
    w = bbox[2]*scalex
    h = bbox[3]*scaley
    return np.asarray([x,y,w,h])

X = np.random.randn(1,400,400,3)
y = np.asarray([
    [100.5,50.6,50,70],
    [250,150,90,20]
])

Nbatch,H,W,C = X.shape
Nfilter = 16
Wfilter = 3
lr = 1e-3

#img = Input(shape=(H,W,C))
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
out = Convolution2D(Nfilter, Wfilter, Wfilter,
    activation='relu', border_mode='same')(out)
out_box = Convolution2D(4, Wfilter, Wfilter,
    activation='relu', border_mode='same')(out)
out_conf = Convolution2D(1, Wfilter, Wfilter,
    activation='sigmoid', border_mode='same')(out)

yhat = sess.run(out_box,feed_dict={img:X})
S = yhat.shape[1]

P = np.zeros((1,S,S,1))
Y = np.zeros(yhat.shape)

for bbox in y:
    best_iou = 0.0
    for i in range(0,S):
        for j in range(0,S):
            bbox_pred_denorm = denormalize_bbox(yhat[0,i,j,:],W,H,S,i,j)
            iou = calc_iou(bbox_pred_denorm, bbox)
            if iou > best_iou:
                best_iou = iou
                best_index = (i,j)
    P[0,best_index[0],best_index[1],0] = 1.0
    Y[0,best_index[0],best_index[1],:] = normalize_bbox(bbox,W,H,S)
