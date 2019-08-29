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

from PIL import Image

import deepMAR_res34 as dm34

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_worker', default=4, type=int,
                    help='num of cpu threads for data')
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

resume_file = ''
if not args.resume == None:
    resume_file = args.resume.split('/')[-1]

### Setting log dirs
# Save settings
save_dir_root = os.path.join( 'save_dir' )
modelName = 'deepMAR_0513'
save_dir = os.path.join(save_dir_root, modelName, 'run_' + datetime.now().strftime('%Y%m%d%H%M%S'))
if not args.resume == None:
    save_dir = save_dir + '_' + args.resume.split('/')[-1]#resume_file
log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
writer = SummaryWriter(log_dir=log_dir)

code_save_dir = os.path.join(save_dir_root, modelName, 'codes')
if not os.path.exists(code_save_dir):
    os.mkdir(code_save_dir)
os.system('rm -rf %s/*'%code_save_dir)
os.system('cp -r scripts %s'%code_save_dir)
me = __file__
os.system('cp %s %s'%(me, code_save_dir))



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
       


def train():

### Load Dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # 3*H*W, [0, 1]
            normalize,
            ])

    dataset = PETADataLoader(listfile='scripts/PETA/train_list.txt', transform=transform)
    data_loader = data.DataLoader(dataset, BATCH_SIZE, num_workers=args.num_worker, shuffle=True)

    #print (dataset.get_classNum())
### Build Model
    net_ = dm34.DeepMAR_res34(dataset.get_classNum())



    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        net_.load_weights(args.resume)

    if args.cuda:
        if len(gpu_ids) > 1:
            net_ = torch.nn.DataParallel(net_, device_ids=gpu_ids).cuda()
        else:
            #device = torch.device('cuda:1')
            #torch.cuda.set_device(gpu_ids[0]) 
            net_ = net_.cuda()
        cudnn.benchmark = True

    optimizer = optim.SGD(net_.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = F.binary_cross_entropy_with_logits
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    net_.train()

    

    # loss counters
    total_loss = 0
    epoch = 0
    print('Loading the dataset...', len(dataset))

    epoch_size = len(dataset) // BATCH_SIZE
    print('Using the specified args:')
    print(args)

    step_index = 0

    # create batch iterator
    batch_iterator = iter(data_loader)

    iter_counter = 0
    for iteration in range( 0, args.max_iter):
        iter_counter += 1
        if iteration != 0 and (iteration % epoch_size == 0):
            # reset epoch loss counters
            writer.add_scalar('data/total_loss_epoch', total_loss/len(dataset), epoch)
            total_loss = 0
            epoch += 1

            if epoch > args.max_epoch:
                break

        if iteration % epoch_size == 0:
            if epoch > args.warm_up:
                scheduler.step(epoch - args.warm_up)
            elif epoch == args.warm_up:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
            else:
                lrr = 1e-4 + (args.lr - 1e-4) * iteration / (epoch_size * args.warm_up)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lrr
                    


        if iter_counter >= len(batch_iterator):
            batch_iterator = iter(data_loader)
            iter_counter = 0

        # load train data
        images, targets = next(batch_iterator)
        #print (images.shape)

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

        # backprop
        optimizer.zero_grad()
        loss_ = criterion(out, targets)
        loss_.backward()
        optimizer.step()
        t1 = time.time()

        # add log
        total_loss += loss_.item()
        #map_loss += 0#loss_map.item()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + '||epoch:' + repr(epoch) + ' || Loss: %.4f ||' % (loss_.item()), end=' ')
            writer.add_scalar('data/total_loss_iter', loss_.item(), iteration)


        ###  save
        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(net_.state_dict(), os.path.join(save_dir, 'models', modelName + '_iter-' + repr(iteration) + '.pth'))
    torch.save(net_.state_dict(), os.path.join(save_dir, 'models', modelName + '_iter-' + repr(iteration) + '.pth'))


if __name__ == '__main__':
    train()
