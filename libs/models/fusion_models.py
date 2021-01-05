import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torchvision import models

from ..utils.utility import mask_iou

def Soft_aggregation(ps, max_obj):
    
    num_objects, H, W = ps.shape
    em = torch.zeros(1, max_obj+1, H, W).to(ps.device)
    em[0, 0, :, :] =  torch.prod(1-ps, dim=0) # bg prob
    em[0,1:num_objects+1, :, :] = ps # obj prob
    em = torch.clamp(em, 1e-7, 1-1e-7)
    logit = torch.log((em /(1-em)))

    return logit

class Fusion(nn.Module):
    def __init__(self, planes):
        super(Fusion, self).__init__()

        self.attention_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes, planes, kernel_size=1),
            nn.Sigmoid()
            )
        
        self.attention_mask = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes, planes, kernel_size=1),
            nn.Sigmoid()
            )

    def forward(self, f_rgb, f_mask):
        assert f_rgb.shape == f_mask.shape, 'rgb feature shape:{} != mask feature shape:{}'.format(f_rgb.shape, f_mask.shape)
        
        # single channel attenton
        attention_rgb = self.attention_rgb(f_rgb)
        attention_mask = self.attention_mask(f_mask)

        f_rgb_after_attention = torch.mul(f_rgb, attention_rgb)
        f_mask_after_attention = torch.mul(f_mask, attention_mask)

        f_fusion = f_rgb_after_attention + f_mask_after_attention

        return f_fusion

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 

class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()

        resnet_rgb = models.resnet34(pretrained=True)
        resnet_mask = models.resnet34(pretrained=True)

        # RGB branch
        self.conv1_rgb = resnet_rgb.conv1
        self.bn1_rgb = resnet_rgb.bn1
        self.relu_rgb = resnet_rgb.relu  # 1/2, 64
        self.maxpool_rgb = resnet_rgb.maxpool

        self.res2_rgb = resnet_rgb.layer1 # 1/4, 64
        self.res3_rgb = resnet_rgb.layer2 # 1/8, 128
        self.res4_rgb = resnet_rgb.layer3 # 1/16, 256

        # mask branch
        self.conv1_mask = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_mask = resnet_mask.bn1
        self.relu_mask = resnet_mask.relu  # 1/2, 64
        self.maxpool_mask = resnet_mask.maxpool

        self.res2_mask = resnet_mask.layer1 # 1/4, 64
        self.res3_mask = resnet_mask.layer2 # 1/8, 128
        self.res4_mask = resnet_mask.layer3 # 1/16, 256

        # fusion block
        self.fusion1 = Fusion(64)
        self.fusion2 = Fusion(128)
        self.fusion3 = Fusion(256)

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m):
        # f = (in_f - self.mean) / self.std
        f = in_f
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim

        # res1
        x = self.conv1_rgb(f)
        y = self.conv1_mask(m)
        x = self.bn1_rgb(x+y)
        y = self.bn1_mask(y)
        c1_x = self.relu_rgb(x)   # 1/2, 64
        c1_y = self.relu_mask(y)   # 1/2, 64
        r1_x = self.maxpool_rgb(c1_x)  # 1/4, 64
        r1_y = self.maxpool_mask(c1_y)  # 1/4, 64

        # res2
        r2_x = self.res2_rgb(r1_x) # 1/4, 64
        r2_y = self.res2_mask(r1_y) # 1/4, 64
        r2_x_fusion = self.fusion1(r2_x, r2_y)

        # res3
        r3_x = self.res3_rgb(r2_x_fusion) # 1/8, 128
        r3_y = self.res3_mask(r2_y) # 1/8, 128
        r3_x_fusion = self.fusion2(r3_x, r3_y)

        # res4
        r4_x = self.res4_rgb(r3_x_fusion) # 1/16, 256
        r4_y = self.res4_mask(r3_y) # 1/16, 256
        r4_x_fusion = self.fusion3(r4_x, r4_y)

        return r4_x_fusion, r3_x_fusion, r2_x_fusion, c1_x
 
class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()

        resnet = models.resnet34(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 64
        self.res3 = resnet.layer2 # 1/8, 128
        self.res4 = resnet.layer3 # 1/16, 256

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f):
        # f = (in_f - self.mean) / self.std
        f = in_f

        x = self.conv1(f) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 64
        r3 = self.res3(r2) # 1/8, 128
        r4 = self.res4(r3) # 1/16, 256

        return r4, r3, r2, c1


class Refine(nn.Module):
    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, size=s.shape[2:], mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, inplane, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(inplane, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(128, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(64, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4, r3, r2, f):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4) # out: 1/8, 256
        m2 = self.RF2(r2, m3) # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        
        p = F.interpolate(p2, size=f.shape[2:], mode='bilinear', align_corners=False)
        return p

class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
 
    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        _, _, H, W = q_in.size()
        no, centers, C = m_in.size()
        _, _, vd = m_out.shape
 
        qi = q_in.view(-1, C, H*W) 
        p = torch.bmm(m_in, qi) # no x centers x hw
        p = p / math.sqrt(C)
        p = torch.softmax(p, dim=1) # no x centers x hw

        mo = m_out.permute(0, 2, 1) # no x c x centers 
        mem = torch.bmm(mo, p) # no x c x hw
        mem = mem.view(no, vd, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p

class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        # self.Key = nn.Linear(indim, keydim)
        # self.Value = nn.Linear(indim, valdim)
        self.Key = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)
 
    def forward(self, x):  
        return self.Key(x), self.Value(x)

class STM(nn.Module):
    def __init__(self, keydim, valdim, phase='test', mode='recurrent', iou_threshold=0.5):
        super(STM, self).__init__()
        self.Encoder_M = Encoder_M() 
        self.Encoder_Q = Encoder_Q()

        self.keydim = keydim
        self.valdim = valdim

        self.KV_M_r4 = KeyValue(256, keydim=keydim, valdim=valdim)
        self.KV_Q_r4 = KeyValue(256, keydim=keydim, valdim=valdim)
        # self.Routine = DynamicRoutine(channel, iters, centers)

        self.Memory = Memory()
        self.Decoder = Decoder(2*valdim, 256)
        self.phase = phase
        self.mode = mode
        self.iou_threshold = iou_threshold

        assert self.phase in ['train', 'test']

    def load_param(self, weight):

        s = self.state_dict()
        for key, val in weight.items():

            # process ckpt from parallel module
            if key[:6] == 'module':
                key = key[7:]

            if key in s and s[key].shape == val.shape:
                s[key][...] = val
            elif key not in s:
                print('ignore weight from not found key {}'.format(key))
            else:
                print('ignore weight of mistached shape in key {}'.format(key))

        self.load_state_dict(s)

    def memorize(self, frame, masks, num_objects): 
        # memorize a frame 
        # maskb = prob[:, :num_objects, :, :]
        # make batch arg list
        frame_batch = []
        mask_batch = []
        # print('\n')
        # print(num_objects)
        try:
            for o in range(1, num_objects+1): # 1 - no
                frame_batch.append(frame)
                mask_batch.append(masks[:,o])

            # make Batch
            frame_batch = torch.cat(frame_batch, dim=0)
            mask_batch = torch.cat(mask_batch, dim=0)
        except RuntimeError as re:
            print(re)
            print(num_objects)
            raise re

        r4, _, _, _ = self.Encoder_M(frame_batch, mask_batch) # no, c, h, w
        _, c, h, w = r4.size()
        memfeat = r4
        # memfeat = self.Routine(memfeat, maskb)
        # memfeat = memfeat.view(-1, c)
        k4, v4 = self.KV_M_r4(memfeat)
        k4 = k4.permute(0, 2, 3, 1).contiguous().view(num_objects, -1, self.keydim)
        v4 = v4.permute(0, 2, 3, 1).contiguous().view(num_objects, -1, self.valdim)
        
        return k4, v4, r4

    def segment(self, frame, keys, values, num_objects, max_obj): 
        # segment one input frame

        r4, r3, r2, _ = self.Encoder_Q(frame)
        n, c, h, w = r4.size()
        # r4 = r4.permute(0, 2, 3, 1).contiguous().view(-1, c)
        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16
        # k4 = k4.view(n, self.keydim, -1).permute(0, 2, 1)
        # v4 = v4.view(n, self.valdim, -1).permute(0, 2, 1)

        # expand to ---  no, c, h, w
        k4e, v4e = k4.expand(num_objects,-1,-1,-1), v4.expand(num_objects,-1,-1,-1) 
        r3e, r2e = r3.expand(num_objects,-1,-1,-1), r2.expand(num_objects,-1,-1,-1)

        m4, _ = self.Memory(keys, values, k4e, v4e)
        logit = self.Decoder(m4, r3e, r2e, frame)
        ps = F.softmax(logit, dim=1)[:, 1] # no, h, w  
        # ps = torch.sigmoid(logit)[:, 1]
        #ps = indipendant possibility to belong to each object
        logit = Soft_aggregation(ps, max_obj) # 1, K, H, W

        return logit, ps

    def forward(self, frame, mask=None, keys=None, values=None, num_objects=None, max_obj=None):

        if self.phase == 'test':
            if mask is not None: # keys
                return self.memorize(frame, mask, num_objects)
            else:
                return self.segment(frame, keys, values, num_objects, max_obj)
        elif self.phase == 'train':

            N, T, C, H, W = frame.size()
            max_obj = mask.shape[2]-1

            total_loss = 0.0
            batch_out = []
            for idx in range(N):

                num_object = num_objects[idx].item()

                batch_keys = []
                batch_vals = []
                tmp_out = []
                for t in range(1, T):
                    # memorize
                    if t-1 == 0 or self.mode == 'mask':
                        tmp_mask = mask[idx, t-1:t]
                    elif self.mode == 'recurrent':
                        tmp_mask = out
                    else:
                        pred_mask = out[0, 1:num_object+1]
                        iou = mask_iou(pred_mask, mask[idx, t-1, 1:num_object+1])

                        if iou > self.iou_threshold:
                            tmp_mask = out
                        else:
                            tmp_mask = mask[idx, t-1:t]

                    key, val, _ = self.memorize(frame=frame[idx, t-1:t], masks=tmp_mask, 
                        num_objects=num_object)

                    batch_keys.append(key)
                    batch_vals.append(val)
                    # segment
                    tmp_key = torch.cat(batch_keys, dim=1)
                    tmp_val = torch.cat(batch_vals, dim=1)
                    logits, ps = self.segment(frame=frame[idx, t:t+1], keys=tmp_key, values=tmp_val, 
                        num_objects=num_object, max_obj=max_obj)

                    out = torch.softmax(logits, dim=1)
                    tmp_out.append(out)
                
                batch_out.append(torch.cat(tmp_out, dim=0))

            batch_out = torch.stack(batch_out, dim=0) # B, T, No, H, W

            return batch_out

        else:
            raise NotImplementedError('unsupported forward mode %s' % self.phase)
