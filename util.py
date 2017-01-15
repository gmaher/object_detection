import numpy as np
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
    if ((xr-xl) <= 0) or ((yb-yt) <=0):
        return 0.0
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
    w = bbox[2]*float(W)
    h = bbox[3]*float(H)
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
    w = float(bbox[2])/W
    h = float(bbox[3])/H
    return np.asarray([x,y,w,h])

def getBboxLabels(yhat,y,W,H,S):
    '''
    computes which predicted boxes in yhat match those in  the true labels y

    args:
        @a yhat: predicted bounding boxes, shape = (Nbatch,S,S,4)
        @a y: true label bounding boxes shape = (Nlabel,4)
        @a W,H,S: image width, height and output dims
    '''
    inds = []
    boxes = []
    for bbox in y:
        best_iou = 0.0
        best_box = 0.0
        best_box_pred = 0.0
        for i in range(0,S):
            for j in range(0,S):
                bbox_pred_denorm = denormalize_bbox(yhat[0,i,j,:],W,H,S,i,j)
                iou = calc_iou(bbox_pred_denorm, bbox)
                if iou >= best_iou:
                    best_iou = iou
                    best_index = (i,j)
                    best_box = bbox
                    best_box_pred = bbox_pred_denorm
        inds.append(best_index)
        boxes.append(best_box)

        # print 'iou={}'.format(best_iou)
        # print 'boxpred={}'.format(best_box_pred)
        # print 'box={}'.format(best_box)
    return boxes,inds

def scalePredictions(prediction, output_shape, S):
    """
    scales the yolo predictions x and y coordinates to be image fractions

    Args:
        prediction (tensor): NbatchxSxSxBx(5+Nclasses) yolo output tensor
        output_shape (tuple): tuple containing prediction output shape
        S (int): Number of grid cells per image dimension

    Returns:
        tensor: NbatchxSxSxBx(5+Nclasses)
    """
    offset = np.zeros(output_shape)
    scale = np.ones(output_shape)

    for i in range(S):
        for j in range(S):
            offset[:,i,j,:,0] = float(j)/S
            offset[:,i,j,:,1] = float(i)/S
            scale[:,i,j,:,:2] = 1.0/S

    return prediction*scale + offset
    
def IOU(prediction,labels):
    """
    calculates the IOU between labels and predicted bounding boxes in each
    grid cell

    Args:
        prediction (tensor): NbatchxSxSxBx4 (x,y,w,h) yolo bounding box tensor, must
        be already scaled so that x and y are represent whole image ratios
        labels (numpy array): NbatchxSxSxBx4 (x,y,w,h) label bounding box array

    Returns:
        tensor: NbatchxSxSxB tensor of IOUs
    """
    pred_xl = prediction[:,:,:,:,0]
    pred_yt = prediction[:,:,:,:,1]
    pred_xr = prediction[:,:,:,:,0]+prediction[:,:,:,:,2]/2.0
    pred_yb = prediction[:,:,:,:,1]+prediction[:,:,:,:,3]/2.0

    lab_xl = labels[:,:,:,:,0]
    lab_yt = labels[:,:,:,:,1]
    lab_xr = labels[:,:,:,:,0]+labels[:,:,:,:,2]/2.0
    lab_yb = labels[:,:,:,:,1]+labels[:,:,:,:,3]/2.0

    xr = tf.select(tf.less_equal(pred_xr,lab_xr), pred_xr, lab_xr)
    xl = tf.select(tf.greater_equal(pred_xl,lab_xl), pred_xl,lab_xl)
    yb = tf.select(tf.less_equal(pred_yb,lab_yb), pred_yb,lab_yb)
    yt = tf.select(tf.greater_equal(pred_yt,lab_yt),pred_yt,lab_yt)

    isct = (xr-xl)*(yb-yt)
    union = prediction[:,:,:,:,2]*prediction[:,:,:,:,3] +\
     labels[:,:,:,:,2]*labels[:,:,:,:,3] - isct

    return isct/(union + EPS)

def yolo_loss(prediction, labels, lam_coord, lam_noobj, S):
    """
    calculates the yolo loss function as a tensorflow tensor

    Args:
        prediction (tensor): NbatchxSxSxBx(5+Nclasses) (x,y,w,h,p) yolo bounding box tensor
        labels (tensor):  NbatchxSxSxBx(5+Nclasses) (x,y,w,h,p) bounding box labels
        lam_coord (float): scaling coefficient for coordinate loss
        lam_noobj (float): scaling coefficient for loss where no object is present
        S (int): number of cells per image axis
    """
