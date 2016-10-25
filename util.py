import numpy as np
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
