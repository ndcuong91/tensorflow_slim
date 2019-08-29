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
import deepMAR as dm

import redis
import pickle


class PersonAttributeRecognition:
    def __init__(self, root, model="weight/deepMAR_0517_iter-926926.pth"):

        # CUDA settings
        gpu_ids = [0]
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        ## build model
        self.net_ = dm.DeepMAR_res50(31)
        self.net_.training = False
        self.net_.load_state_dict(torch.load(os.path.join(root, model),
                                        map_location=lambda storage, loc: storage))
        self.net_ = self.net_.cuda()

        self.class_list = []
        cfile = open(os.path.join(root, 'scripts/PETA/peta_valid_labels_v2.txt'),'r')
        for line in cfile:
            self.class_list.append(line.rstrip('\n'))


    def shaping_results(self, ped_result, out):
        """
        age
        gender
        carrying
        hair
        lowerClothes
        upperClothes
        accessory
        """
        age_labels = [0,4]
        gen_labels = [4,6]
        cary_labels = [6,9]
        hair_labels = [9,10]
        lowCloth_labels = [11,15]
        upCloth_labels = [15,21]
        lowColor_labels = [24,27]
        upColor_labels = [27,31]

        shap_result = {}
        
        shap_result['Age'] = self.class_list[np.argmax(out[age_labels[0]:age_labels[1]])]
        shap_result['Age'] = shap_result['Age'].replace('personal','')

        shap_result['Gender'] = self.class_list[np.argmax(out[gen_labels[0]:gen_labels[1]]) + gen_labels[0]]
        shap_result['Gender'] = shap_result['Gender'].replace('personal','')

        shap_result['Carrying'] = self.class_list[np.argmax(out[cary_labels[0]:cary_labels[1]]) + cary_labels[0]]
        shap_result['Carrying'] = shap_result['Carrying'].replace('carrying','')

        shap_result['Hair'] = self.class_list[np.argmax(out[hair_labels[0]:hair_labels[1]]) + hair_labels[0]]
        shap_result['Hair'] = shap_result['Hair'].replace('hair','')

        lctype = self.class_list[np.argmax(out[lowCloth_labels[0]:lowCloth_labels[1]]) + lowCloth_labels[0]].replace('lowerBody','')
        lccolor = self.class_list[np.argmax(out[lowColor_labels[0]:lowColor_labels[1]]) + lowColor_labels[0]].replace('lowerBody','')
        shap_result['LowerClothes'] = lccolor + '_' + lctype

        uctype = self.class_list[np.argmax(out[upCloth_labels[0]:upCloth_labels[1]]) + upCloth_labels[0]].replace('upperBody','')
        uccolor = self.class_list[np.argmax(out[upColor_labels[0]:upColor_labels[1]]) + upColor_labels[0]].replace('upperBody','')
        shap_result['UpperClothes'] = uccolor + '_' + uctype

        return shap_result



    def run_cv2Img(self, decoded_img, det_result):
        h, w, ch = decoded_img.shape

        result = {}
        result['results'] = []
        for dets in det_result:
            if dets['class'] != 'person':
                continue
        
            ped_result = {}
            x1 = int(dets['bbox']['x1'] * w)
            x2 = int(dets['bbox']['x2'] * w)
            y1 = int(dets['bbox']['y1'] * h)
            y2 = int(dets['bbox']['y2'] * h)

            img = decoded_img[y1:y2, x1:x2]
            if img.shape[0] < 10 or img.shape[1] < 10:
                continue
            p_img = cv2.resize(img, (224, 224)).astype(np.float32)
            p_img /= 255.0
            p_img -= (0.485, 0.456, 0.406)
            p_img /= (0.229, 0.224, 0.225)
            x = p_img[:, :, ::-1].copy()
            x = torch.from_numpy(x).permute(2, 0, 1)
            xx = Variable(x.unsqueeze(0))
            xx = xx.cuda()

            out = self.net_(xx)
            out = torch.sigmoid(out)[0].data.cpu().numpy()

            for i in range(len(self.class_list)):
                ped_result[self.class_list[i]] = float(out[i])
            
            shaped_result = self.shaping_results(ped_result, out)
            
            result['results'].append(shaped_result)
        return result

