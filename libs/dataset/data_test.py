from __future__ import division
import sys
cv2_WRONG_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if cv2_WRONG_PATH in sys.path:
    sys.path.remove(cv2_WRONG_PATH)

from options import OPTION as opt
from libs.dataset.transform import TrainTransform, TestTransform
from libs.dataset.data import DATA_CONTAINER, multibatch_collate_fn, convert_one_hot
from libs.dataset.image_data import COCODataset
import torch.utils.data as data

import matplotlib.pyplot as plt
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

input_dim = opt.input_size
train_transformer = TrainTransform(size=input_dim)

ds = COCODataset(transform=train_transformer)
print(len(ds))
trainloader = data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=1,
                                  collate_fn=multibatch_collate_fn, drop_last=True)
num = 4
plt.figure()
for i, data in enumerate(trainloader):
    if i == num:
        break
    frames, masks, num_objs, info = data
    print(frames.shape, masks.shape, num_objs.shape)

    frame = frames[0]
    mask = masks[0]
    num_obj = num_objs[0]
    for j in range(frame.shape[0]):
        ax = plt.subplot(2*num, 3, i*6+j+1)
        ax.axis('off')
        ax.imshow(frame[j].numpy().transpose(1, 2, 0))
        plt.pause(0.01)
    for k in range(mask.shape[0]):
        ax = plt.subplot(2*num, 3, i*6+4+k)
        ax.axis('off')
        # ax.imshow(np.array(mask[k, 0], dtype=np.uint8))
        ax.imshow(convert_one_hot(np.array(mask[k],dtype=np.uint8).transpose(1, 2, 0), num_obj.item()))
        plt.pause(0.01)
plt.show()
