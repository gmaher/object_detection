from util import *

W = 400
H = 400
S = 200
box = [100,50.5,20,20]
box_test = [100,50.5,10,10]
box_norm = [0,0.25,0.05,0.05]

print '{} normalizes to {}'.format(box, normalize_bbox(box,W,H,S))
print '{} denormalizes to {}'.format(box_norm, denormalize_bbox(box_norm,W,H,S,25,50))
print 'IOU of {} with {} = {}'.format(box,box,calc_iou(box,box))
print 'IOU of {} with {} = {}'.format(box,box_test,calc_iou(box,box_test))
