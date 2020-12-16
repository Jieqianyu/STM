from libs.dataset.data import DATA_CONTAINER, multibatch_collate_fn
from libs.dataset.transform import TrainTransform, TestTransform
from libs.utils.logger import Logger, AverageMeter
from libs.utils.loss import *
from libs.utils.utility import write_mask, save_checkpoint, adjust_learning_rate, mask_iou
from libs.models.models import STM

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import os
import os.path as osp
import shutil
import time
import pickle
from progress.bar import Bar
from collections import OrderedDict

from options import OPTION as opt

MAX_FLT = 1e6

# Use CUDA
device = 'cuda:{}'.format(opt.gpu_id)
use_gpu = torch.cuda.is_available() and int(opt.gpu_id) >= 0

def main():

    # Data
    print('==> Preparing dataset %s' % opt.valset)

    input_dim = opt.input_size

    test_transformer = TestTransform(size=input_dim)

    testset = DATA_CONTAINER[opt.valset](
        train=False, 
        transform=test_transformer, 
        samples_per_video=1
        )

    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                 collate_fn=multibatch_collate_fn)
    # Model
    print("==> creating model")

    net = STM(opt.keydim, opt.valdim)
    print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

    # set eval to freeze batchnorm update
    net.eval()

    if use_gpu:
        net.to(device)

    # set testing parameters
    for p in net.parameters():
        p.requires_grad = False

    # Strateges
    criterion = None
    celoss = cross_entropy_loss

    if opt.loss == 'ce':
        criterion = celoss
    elif opt.loss == 'iou':
        criterion = mask_iou_loss
    elif opt.loss == 'both':
        criterion = lambda pred, target, obj: celoss(pred, target, obj) + mask_iou_loss(pred, target, obj)
    else:
        raise TypeError('unknown training loss %s' % opt.loss)

    # Resume
    title = 'STM'

    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint {}'.format(opt.resume))
        assert os.path.isfile(opt.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.resume, map_location=device)
        state = checkpoint['state_dict']
        net.load_param(state)

    # Train and val
    print('==> Runing model on dataset {}, totally {:d} videos'.format(opt.valset, len(testloader)))

    test(testloader,
        model=net,
        criterion=criterion,
        use_cuda=use_gpu,
        opt=opt)

    print('==> Results are saved at: {}'.format(os.path.join(opt.results, opt.valset))) 
        

def test(testloader, model, criterion, use_cuda, opt):

    data_time = AverageMeter()
    fps = AverageMeter()

    for batch_idx, data in enumerate(testloader):
        frames, masks, objs, infos = data

        if use_cuda:
            frames = frames.to(device)
            masks = masks.to(device)
                
        frames = frames[0]
        masks = masks[0]
        num_objects = objs[0]
        info = infos[0]
        max_obj = masks.shape[1]-1
        T, _, H, W = frames.shape

        bar = Bar(info['name'], max=T-1)
        print('==>Runing video {}, objects {:d}'.format(info['name'], num_objects))
        # compute output
            
        pred = [masks[0:1]]
        keys = []
        vals = []
        for t in range(1, T):
            if t-1 == 0:
                tmp_mask = masks[0:1]
            elif 'frame' in info and t-1 in info['frame']:
                # start frame
                mask_id = info['frame'].index(t-1)
                tmp_mask = masks[mask_id:mask_id+1]
                num_objects = max(num_objects, tmp_mask.max())
            else:
                tmp_mask = out

            t1 = time.time()
            # memorize
            key, val, _ = model(frame=frames[t-1:t, :, :, :], mask=tmp_mask, num_objects=num_objects)

            # segment
            tmp_key = torch.cat(keys+[key], dim=1)
            tmp_val = torch.cat(vals+[val], dim=1)
            logits, ps = model(frame=frames[t:t+1, :, :, :], keys=tmp_key, values=tmp_val, num_objects=num_objects, max_obj=max_obj)

            out = torch.softmax(logits, dim=1)

            # # gradient correction process
            # if t % opt.save_freq == 0:
            #     pred.append(out.clone())
            #     # track gradient
            #     out.requires_grad = True
            #     optimizer = optim.SGD([out], lr=opt.correction_lr)

            #     for i in range(opt.correction_iter_times):
            #         # memorize current frame
            #         t_key, t_val, _ = model(frame=frames[t:t+1, :, :, :], mask=out, num_objects=num_objects)
            #         # segment the first frame
            #         t_logits, _ = model(frame=frames[0:1, :, :, :], keys=t_key, values=t_val, num_objects=num_objects, max_obj=max_obj)
            #         out0 = torch.softmax(t_logits, dim=1)
            #         # loss
            #         loss = criterion(out0, masks[0:1], num_objects)
            #         # update out
            #         optimizer.zero_grad()
            #         loss.backward()
            #         optimizer.step()
                
            #     # no tracking gradient
            #     out.requires_grad = False
            # else:
            #     pred.append(out)
            pred.append(out)

            if (t-1) % opt.save_freq == 0:
                keys.append(key)
                vals.append(val)
            # _, idx = torch.max(out, dim=1)

            toc = time.time() - t1

            data_time.update(toc, 1)
            fps.update(1/toc, 1)

            # plot progress
            bar.suffix  = '({batch}/{size}) Average Fps: {fps:.1f} Cumulative Time: {data:.3f}s'.format(
                batch=t,
                size=T-1,
                fps=fps.avg,
                data=data_time.sum
            )
            bar.next()
        bar.finish()
            
        pred = torch.cat(pred, dim=0)
        pred = pred.detach().cpu().numpy()
        # write_mask(pred, info, opt)     
           

    return

if __name__ == '__main__':
    main()
