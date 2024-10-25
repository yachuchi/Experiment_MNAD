import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from model.AE import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob

import argparse


parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--method', type=str, default='ae', help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=1, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')

parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='shanghai', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
parser.add_argument('--model_dir', type=str, help='directory of model')
parser.add_argument('--exp_dir', type=str, default='evaluate_log', help='directory of log')


args = parser.parse_args()

log_dir = os.path.join('./exp', args.dataset_type, args.method, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'),'w')
sys.stdout= f

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"

# Loading dataset
test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model
model = torch.load(args.model_dir)
model.cuda()

labels = np.load('./data/frame_labels_'+args.dataset_type+'.npy')

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
for video in videos_list:
    video_name = video.split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
psnr_list = {}
mse_list = {}
feature_distance_list = {}

print('Evaluation of', args.dataset_type)

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    if args.method == 'ae':
        #shanghaitec
        labels_list = np.append(labels_list, labels[label_length:videos[video_name]['length']+label_length])
        #ped2 & cuhk
        # labels_list = np.append(labels_list, labels[0][label_length:videos[video_name]['length']+label_length])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    mse_list[video_name] = []
    feature_distance_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]]['length']


model.eval()

for k,(imgs) in enumerate(test_batch):
    
    if args.method == 'ae':
        if k == label_length:
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    imgs = Variable(imgs).cuda()
    
    if args.method == 'ae':
        outputs = model.forward(imgs,False)
        # mse_imgs = torch.mean(loss_func_mse(outputs, imgs)).item()
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()
        
    # outputs = model.forward(imgs,False)
    # mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()    
    
    mse_list[videos_list[video_num].split('/')[-1]].append(mse_imgs)
    # psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
# print(f'{mse_list}')

# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    anomaly_score_total_list += mse_list[video_name]

anomaly_score_total_list = np.asarray(anomaly_score_total_list)

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

print('The result of ', args.dataset_type)
print('AUC: ', accuracy*100, '%')
