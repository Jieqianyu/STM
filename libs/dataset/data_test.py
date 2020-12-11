from __future__ import division
import sys
cv2_WRONG_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if cv2_WRONG_PATH in sys.path:
    sys.path.remove(cv2_WRONG_PATH)

from options import OPTION as opt
from libs.dataset.transform import TrainTransform, TestTransform
from libs.dataset.data import DATA_CONTAINER, multibatch_collate_fn
from libs.dataset.image_data import COCODataset
import torch.utils.data as data

import cv2
import numpy as np
import torch


print('==> Preparing dataset')

input_dim = opt.input_size
train_transformer = TrainTransform(size=input_dim)

# ds = DATA_CONTAINER['DAVIS17'](
#                 train=True, 
#                 sampled_frames=opt.sampled_frames, 
#                 transform=train_transformer, 
#                 max_skip=opt.max_skip[0], 
#                 samples_per_video=opt.samples_per_video
#             )

# trainloader = data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=opt.workers,
#                               collate_fn=multibatch_collate_fn, drop_last=True)

# _, (frame, mask, num_obj, info) = next(enumerate(trainloader))
# print("frame:", type(frame), frame.shape)
# print("mask:", type(mask), mask.shape)
# print("num_obj:", type(num_obj), num_obj)
# print("info:", type(info))

ds = COCODataset()
trainloader = data.DataLoader(ds, batch_size=1, num_workers=2)
it = enumerate(trainloader)
for j in range(10):
    sequence_data, num = next(it)
print(type(sequence_data), num)
#print(sequence_data['image'], torch.max(sequence_data['anno'][1]))

for i in range(len(sequence_data)):
    cv2.imwrite('/home/jm/test_{}.png'.format(str(i)), np.array(sequence_data[i][0].numpy()*255, dtype=np.uint8))