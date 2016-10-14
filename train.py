import numpy as np
from keras.layers import Input, Convolution2D, MaxPooling2D
from keras.objectives import mean_absolute_error
from keras.models import Model
from keras.optimizers import Adam

def calc_iou(box_pred,box_true):
    '''
    function to calculate intersection over union of two bounding boxes

    args:
        @a box_pred, box_true [x,y,w,h]
    '''
    Apred = box_pred[2]*box_pred[3]
    Atrue = box_true[2]*box_true[3]
    
X = np.random.randn(1,400,400,3)
y = np.asarray([
    [100,50,50,70],
    [250,150,90,20]
])

Nbatch,H,W,C = X.shape
Nfilter = 16
Wfilter = 3
lr = 1e-3

img = Input(shape=(H,W,C))

out = MaxPooling2D(pool_size=(2,2))(img)
out = Convolution2D(Nfilter, Wfilter, Wfilter,
    activation='relu', border_mode='same')(out)
out = Convolution2D(Nfilter, Wfilter, Wfilter,
    activation='relu', border_mode='same')(out)
out = Convolution2D(Nfilter, Wfilter, Wfilter,
    activation='relu', border_mode='same')(out)
out = Convolution2D(4, Wfilter, Wfilter,
    activation='linear', border_mode='same')(out)

adam = Adam(lr)

net = Model(img,out)
net.compile(optimizer=adam,
    loss='mean_absolute_error')

yhat = net.predict(X)

P = np.zeros(yhat.shape)

S = yhat.shape[0]
for bbox in y
    best_iou = 0.0
    for i in range(0,S):
        for j in range(0,S):
            iou = calc_iou(yhat[i,j,:],bbox)
            if iou > best_iou:
                best_iou = iou
                best_index = (i,j)
    P[0,best_index[0],best_index[1],:] = bbox

net.train_on_batch(X,P)
