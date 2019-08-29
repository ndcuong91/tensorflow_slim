# -*- coding: utf-8 -*-
import argparse
import glob
import os
import sys
import time

import numpy as np
# PyTorch includes
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

import cv2
import deepMAR_res34 as dm34

import redis
import pickle

#### IO (Redis)
# https://qiita.com/FukuharaYohei/items/48209d488bc7f412c3d7
host = "127.0.0.1"
port = 6379
pool = redis.ConnectionPool(host=host, port=port, db=0)
redis_con = redis.StrictRedis(connection_pool=pool)


# CUDA settings
gpu_ids = [0]
os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

## build model
net_ = dm34.DeepMAR_res34(74)
net_.load_state_dict(torch.load("weight/deepMAR_0513_iter-93526.pth",
                                 map_location=lambda storage, loc: storage))
net_ = net_.cuda()

class_list = []
cfile = open('scripts/PETA/peta_valid_labels.txt','r')
for line in cfile:
    class_list.append(line.rstrip('\n'))

while True:
    modpose_tstmp = redis_con.lindex('tstmp:modpose:camera:1', 0)
    modpose_data = redis_con.lindex('modpose:camera:1', 0)

    camera_tstmp = redis_con.lrange('tstmp:camera:1', 0, -1)
    cam_index = camera_tstmp.index(modpose_tstmp)
    camera_data = redis_con.lindex('camera:1', cam_index)

    #if isinstance(camera_data, bytes):
    decoded_img = cv2.imdecode(np.frombuffer(camera_data, np.uint8), 1)
    det_result = pickle.loads(modpose_data)['det_results']
    
    h, w, ch = decoded_img.shape
    for dets in det_result:
        if dets['class'] != 'person':
            continue
        
        x1 = int(dets['bbox']['x1'] * w)
        x2 = int(dets['bbox']['x2'] * w)
        y1 = int(dets['bbox']['y1'] * h)
        y2 = int(dets['bbox']['y2'] * h)

        #decoded_img = cv2.rectangle(decoded_img, (x1,y1), (x2,y2), (255,0,0), 3)
        img = decoded_img[y1:y2, x1:x2]
        p_img = cv2.resize(img, (224, 224)).astype(np.float32)
        p_img /= 255.0
        p_img -= (0.485, 0.456, 0.406)
        p_img /= (0.229, 0.224, 0.225)
        x = p_img[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = Variable(x.unsqueeze(0))
        xx = xx.cuda()

        out = net_(xx)
        out = F.sigmoid(out)[0].data.cpu().numpy()

        for i in range(len(class_list)):
            print (class_list[i], '\t' ,out[i])

    cv2.imshow("res", img)
    cv2.waitKey(-1)
    


