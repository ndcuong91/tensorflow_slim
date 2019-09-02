# -*- coding: utf-8 -*-
import os
import glob
import sys
import time

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import argparse

import torchvision 

# Tensorboard include
from tensorboardX import SummaryWriter
from datetime import datetime
import socket
from torch.optim import lr_scheduler
from torchsummary import summary

from PIL import Image
import cv2

import deepMAR as dm

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')

parser.add_argument('--max_iter', default=1000000, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--max_epoch', default=100, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--warm_up', default=1e-4, type=float,
                    help='Gamma update for SGD')             


                    
args = parser.parse_args()


BATCH_SIZE = args.batch_size
gpu_ids = [0]
#os.system('export CUDA_VISIBLE_DEVICES=1')
os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
#print (os.environ['CUDA_VISIBLE_DEVICES'])
#exit()



# CUDA settings
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class PETADataLoader(torch.utils.data.Dataset):
    def __init__(self, listfile, transform=None):
        self.transform = transform
        self.imgList = []
        self.dataList = []

        lf = open(listfile, 'r')
        for line in lf:
            sep = line.rstrip('\n').split(',')
            img_path = sep[0]
            ldata = sep[1:]
            ldata = list(map(int, ldata))
            self.imgList.append(img_path)
            self.dataList.append(ldata)

    def __getitem__(self, index):
        imgpath = self.imgList[index]
        target = self.dataList[index]

        img = Image.open(imgpath)
        if self.transform is not None:
            img = self.transform( img )

        target = np.array(target).astype(np.float32)

        return img, target


    def __len__(self):
        return len(self.imgList)
    
    def get_classNum(self):
        return len(self.dataList[0])

def calculate_mAP(y_pred, y_true, num_class=31):
    average_precisions = []

    for index in range(num_class):
        pred = y_pred[:, index]
        label = y_true[:, index]

        sorted_indices = np.argsort(-pred)
        sorted_label = label[sorted_indices]

        tp = (sorted_label == 1)
        fp = (sorted_label == 0)

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        npos = np.sum(sorted_label)
        recall = tp * 1.0 / npos

        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp * 1.0 / np.maximum((tp + fp), np.finfo(np.float64).eps)

        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        AP=np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        print('Class ', index, ':', round(AP,3) )
        average_precisions.append(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))

    #print(average_precisions)
    mAP = np.mean(average_precisions)

    return mAP


def test_resnet34(pretrained_model, num_class = 74, total_samples = 1376):

### Load Dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # 3*H*W, [0, 1]
            normalize,
            ])

    dataset = PETADataLoader(listfile='scripts/PETA/val_list.txt', transform=transform)
    data_loader = data.DataLoader(dataset, 1, num_workers=1, shuffle=True)

### Build Model
    net_ = dm.DeepMAR_res34(dataset.get_classNum())
    #print(net_)
    summary(net_, (3, 224, 224))
    net_.load_state_dict(torch.load(pretrained_model, map_location=lambda storage, loc: storage))

    if args.cuda:
        if len(gpu_ids) > 1:
            net_ = torch.nn.DataParallel(net_, device_ids=gpu_ids).cuda()
        else:
            net_ = net_.cuda()
        cudnn.benchmark = True

    batch_iterator = iter(data_loader)

    # CuongND evaluation
    true_preds = []
    for i in range(num_class):
        true_preds.append(0)

    for iteration in range( 0, total_samples):
        if(iteration%100==0 and iteration>0):
            print ('Predict:',iteration,'images')

        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = Variable(targets.cuda())
                #targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            with torch.no_grad():
                targets = Variable(targets)
                #targets = [Variable(ann) for ann in targets]

        # forward
        t0 = time.time()
        out = net_(images)
        out = F.sigmoid(out)


        for i, tval in enumerate(targets[0]):
            true_val=tval.item()
            pred_val=out[0][i].item()

            if(pred_val<0.5):
                pred_val=0
            elif(pred_val>0.5):
                pred_val=1
            else:
                true_preds[i]+=1
                continue
            if(pred_val==int(true_val)):
                true_preds[i]+=1

        cv2.waitKey(100)

    accum_acc=0
    for i in range(num_class):
        acc = float(true_preds[i]) / float(total_samples)
        accum_acc+=acc
        print('Class ', i, ':', acc)
    print('Final acc:',accum_acc/float(num_class))


def test_resnet50(pretrained_model, num_class = 31, total_samples = 1376):

### Load Dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # 3*H*W, [0, 1]
            normalize,
            ])

    dataset = PETADataLoader(listfile='/home/duycuong/PycharmProjects/research_py3/tensorflow_slim/data/PETA/val_list_v2.txt', transform=transform)
    data_loader = data.DataLoader(dataset, 1, num_workers=1, shuffle=True)

### Build Model
    net_ = dm.DeepMAR_res50(dataset.get_classNum(),pretrained=False)
    summary(net_, (3, 224, 224))
    net_.load_state_dict(torch.load(pretrained_model, map_location=lambda storage, loc: storage))

    if args.cuda:
        if len(gpu_ids) > 1:
            net_ = torch.nn.DataParallel(net_, device_ids=gpu_ids).cuda()
        else:
            net_ = net_.cuda()
        cudnn.benchmark = True

    batch_iterator = iter(data_loader)

    # CuongND evaluation

    true_preds = []
    for i in range(num_class):
        true_preds.append(0)


    prediction_arr=[]
    label_arr=[]

    for iteration in range( 0, total_samples):
        if(iteration%100==0 and iteration>0):
            print ('Predict:',iteration,'images')

        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = Variable(targets.cuda())
                #targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            with torch.no_grad():
                targets = Variable(targets)
                #targets = [Variable(ann) for ann in targets]

        # forward
        t0 = time.time()
        out = net_(images)
        out = torch.sigmoid(out)

        prediction_arr.append(out.cpu().detach().numpy()[0])
        label_arr.append(targets.cpu().numpy()[0])

        for i, tval in enumerate(targets[0]):
            true_val=tval.item()
            pred_val=out[0][i].item()

            if(pred_val<0.5):
                pred_val=0
            elif(pred_val>0.5):
                pred_val=1
            else:
                true_preds[i]+=1
                continue
            if(pred_val==int(true_val)):
                true_preds[i]+=1

        cv2.waitKey(100)


    result = calculate_mAP(np.asarray(prediction_arr), np.asarray(label_arr))
    print('mAP score: {}'.format(result))

    accum_acc=0
    for i in range(num_class):
        acc = float(true_preds[i]) / float(total_samples)
        accum_acc+=acc
        print('Class ', i, ':', round(acc,3))
    print('Final acc:',accum_acc/float(num_class))


if __name__ == '__main__':
    #test_resnet34('save_dir/deepMAR_0513/run_20190821140616/models/deepMAR_0513_iter-93526.pth')
    test_resnet50('save_dir/deepMAR_0517/run_20190828160054/deepMAR_0517_iter-926926.pth')
