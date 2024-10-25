
import os
import sys
import cv2
import math
import glob
import copy
import time
import random
import torch
import torchvision
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model.utils import DataLoader
import torchvision.utils as v_utils
import torchvision.datasets as dset
from torch.autograd import Variable
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import torch.utils.data.dataset as dataset
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
#########################################################################################################################
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from model.Reconstruction import *
from utils import *


import argparse

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--method', type=str, default='recon', help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=1, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='shanghai', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
parser.add_argument('--model_dir', type=str, help='directory of model')
parser.add_argument('--m_items_dir', type=str, help='directory of model')
parser.add_argument('--exp_dir', type=str, default='evaluate_log', help='directory of log')
parser.add_argument('--figure_dir', type=str, default='./figures/', help='directory of log')
args = parser.parse_args()

misjudge_figure_path = args.figure_dir + args.dataset_type + '_misjudge'
correct_figure_path = args.figure_dir + args.dataset_type + '_correct'
heatmap_path = args.figure_dir + args.dataset_type + '_heatmap'
reconstructed_path = args.figure_dir + args.dataset_type + '_reconstruction'
combination_path = args.figure_dir + args.dataset_type + '_output'
# anomaly_figure_path = args.figure_dir + args.dataset_type + '_anomalymap'
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

train_folder = args.dataset_path+"/"+args.dataset_type+"/training/no_background"
test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/no_background"

invTrans = transforms.Compose([
    transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
])
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
# Report the evaluate result
log_dir = os.path.join('./exp', args.dataset_type, args.method, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'),'w')
sys.stdout= f

# Loading test dataset
test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

train_dataset = DataLoader(train_folder, transforms.Compose([
             transforms.ToTensor(),          
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

train_size = len(train_dataset)

train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True)

loss_func_mse = nn.MSELoss(reduction='none')

train_loss_func_mse = nn.MSELoss(reduction='none')
#################################################################################################################################
#
#   Threshold based on training data
#
#################################################################################################################################
# Loading the trained model
model = torch.load(args.model_dir)
model.cuda()
m_items = torch.load(args.m_items_dir)
# train_labels = np.zeros()

videos = OrderedDict()
train_list = sorted(glob.glob(os.path.join(train_folder, '*')))
for video in train_list:
    video_name = video.split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.png'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}

print('Evaluation of', args.dataset_type)

# Setting for video anomaly detection
for video in sorted(train_list):
    video_name = video.split('/')[-1]
    # if args.method == 'pred':
    #     labels_list = np.append(labels_list, train_labels[0][4+label_length:videos[video_name]['length']+label_length])
    # else:
    #     labels_list = np.append(labels_list, train_labels[0][label_length:videos[video_name]['length']+label_length])
    # label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[train_list[video_num].split('/')[-1]]['length']
m_items_test = m_items.clone()

model.eval()
for k,(imgs) in enumerate(train_batch):
    
    if args.method == 'pred':
        if k == label_length-4*(video_num+1):
            video_num += 1
            label_length += videos[train_list[video_num].split('/')[-1]]['length']
    else:
        if k == label_length:
            video_num += 1
            label_length += videos[train_list[video_num].split('/')[-1]]['length']

    imgs = Variable(imgs).cuda()
    
    if args.method == 'pred':
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
        mse_imgs = torch.mean(train_loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs[:,3*4:])
    
    else:
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(imgs, m_items_test, False)
        mse_imgs = torch.mean(train_loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()
        mse_feas = compactness_loss.item()


    psnr_list[train_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
    feature_distance_list[train_list[video_num].split('/')[-1]].append(mse_feas)
    # output_img.append(outputs)


# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
for video in sorted(train_list):
    video_name = video.split('/')[-1]
    anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
                                     anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)

train_anomaly_score_total_list = np.asarray(anomaly_score_total_list)


train_avg_anomaly_score = np.mean(train_anomaly_score_total_list)
train_std_anomaly_score = np.std(train_anomaly_score_total_list)

print(f'train avg anomaly score: {train_avg_anomaly_score}')
print(f'std anomaly score: {train_std_anomaly_score}')
print(f'avg + std anomaly score: {train_avg_anomaly_score + train_std_anomaly_score}')

#################################################################################################################################
#
#   Origin part
#
#################################################################################################################################
# Loading the trained model
# model = torch.load(args.model_dir)
# model.cuda()
# m_items = torch.load(args.m_items_dir)
labels = np.load('./data/frame_labels_'+args.dataset_type+'.npy')

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
for video in videos_list:
    video_name = video.split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.png'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}

print('Evaluation of', args.dataset_type)

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    if args.method == 'pred':
        labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
    else:
        labels_list = np.append(labels_list, labels[label_length:videos[video_name]['length']+label_length])
        # labels_list = np.append(labels_list, labels[0][label_length:videos[video_name]['length']+label_length])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]]['length']
m_items_test = m_items.clone()
mse_list = []
model.eval()
for k,(imgs) in enumerate(test_batch):
    
    if args.method == 'pred':
        if k == label_length-4*(video_num+1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    else:
        if k == label_length:
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    imgs = Variable(imgs).cuda()
    
    if args.method == 'pred':
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs[:,3*4:])
    
    else:
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(imgs, m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs)

    if  point_sc < args.th:
        query = F.normalize(feas, dim=1)
        query = query.permute(0,2,3,1) # b X h X w X d
        m_items_test = model.memory.update(query, m_items_test, False)

    psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
    feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)
    
    # output_img.append(outputs)


# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
                                     anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)

anomaly_score_total_list = np.asarray(anomaly_score_total_list)


accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))


print('The result of ', args.dataset_type)
print('AUC: ', accuracy*100, '%')


#################################################################################################################################
#
#   visualization
#
#################################################################################################################################

os.makedirs(misjudge_figure_path, exist_ok=True)
os.makedirs(correct_figure_path, exist_ok=True)
os.makedirs(heatmap_path, exist_ok=True)
os.makedirs(reconstructed_path, exist_ok=True)
os.makedirs(combination_path, exist_ok=True)
# os.makedirs(anomaly_figure_path, exist_ok=True)


label_0 = 0
label_1 = 0

st0_1 = 0
st1_2 = 0
st2_3 = 0
st3_4 = 0
st4_5 = 0
st5_6 = 0
st6_7 = 0
st7_8 = 0
st8_9 = 0
st9_10 = 0
for idx, (imgs) in enumerate(tqdm(test_batch)):
    
    # if args.method == 'pred':
    #     if idx == label_length-4*(video_num+1):
    #         video_num += 1
    #         label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    # else:
    #     if idx == label_length:
    #         video_num += 1
    #         label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    # imgs = Variable(imgs).cuda()
    
    # if args.method == 'pred':
    #     outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
    #     mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
    #     mse_feas = compactness_loss.item()

    #     # Calculating the threshold for updating at the test time
    #     point_sc = point_score(outputs, imgs[:,3*4:])
    
    # else:
    #     outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(imgs, m_items_test, False)
    #     mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()
    #     mse_feas = compactness_loss.item()

    # psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))

  

    if labels_list[idx] == 0:
        label_0 +=1
    if labels_list[idx] == 1:
        label_1 +=1

    if anomaly_score_total_list[idx]<= 0.1:
        st0_1 +=1
    elif anomaly_score_total_list[idx]> 0.1 and anomaly_score_total_list[idx]<= 0.2:
        st1_2 +=1
    elif anomaly_score_total_list[idx]> 0.2 and anomaly_score_total_list[idx]<= 0.3:
        st2_3 +=1
    elif anomaly_score_total_list[idx]> 0.3 and anomaly_score_total_list[idx]<= 0.4:
        st3_4 +=1
    elif anomaly_score_total_list[idx]> 0.4 and anomaly_score_total_list[idx]<= 0.5:
        st4_5 +=1
    elif anomaly_score_total_list[idx]> 0.5 and anomaly_score_total_list[idx]<= 0.6:
        st5_6 +=1
    elif anomaly_score_total_list[idx]> 0.6 and anomaly_score_total_list[idx]<= 0.7:
        st6_7 +=1
    elif anomaly_score_total_list[idx]> 0.7 and anomaly_score_total_list[idx]<= 0.8:
        st7_8 +=1
    elif anomaly_score_total_list[idx]> 0.8 and anomaly_score_total_list[idx]<= 0.9:
        st8_9 +=1
    else:
        st9_10 +=1
print('------------------------------------------------')
print(f'label 0: {label_0}')
print(f'label 1: {label_1}')
print('------------------------------------------------')
print(f'st0_1={st0_1}')
print(f'st1_2={st1_2}')
print(f'st2_3={st2_3}')
print(f'st3_4={st3_4}')
print(f'st4_5={st4_5}')
print(f'st5_6={st5_6}')
print(f'st6_7={st6_7}')
print(f'st7_8={st7_8}')
print(f'st8_9={st8_9}')
print(f'st9_10={st9_10}')

threshold_1 = train_avg_anomaly_score
threshold_2 = train_avg_anomaly_score + 0.5 * train_std_anomaly_score
threshold_3 = train_avg_anomaly_score + 1* train_std_anomaly_score
threshold_4 = train_avg_anomaly_score - 0.5 * train_std_anomaly_score
threshold_5 = train_avg_anomaly_score - 1 * train_std_anomaly_score
threshold_6 = 0.95
threshold_7 = 0.05

misjudge_GT0_AS1 = 0
misjudge_GT1_AS0 = 0
correct_GT1_AS1 = 0
correct_GT0_AS0 = 0
for idx, (imgs) in enumerate(tqdm(test_batch)):
    if args.method == 'pred':
        if idx == label_length-4*(video_num+1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    else:
        if idx == label_length:
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    imgs = Variable(imgs).cuda()
    
    if args.method == 'pred':
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs[:,3*4:])
    
    else:
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(imgs, m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()
        mse_feas = compactness_loss.item()

    psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))

# anomaly_score_total_list = []
# for video in sorted(videos_list):
#     video_name = video.split('/')[-1]
#     anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
#                                      anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)
#     heatmap = torch.sum(anomaly_score_total_list, dim=1)
#     max_value = torch.max(heatmap)
#     min_value = torch.min(heatmap)
#     heatmap = (heatmap - min_value)/(heatmap - max_value)*255
#     heatmap = heatmap.detach().cpu().numpy()
#     heatmap = heatmap.astype(np.uint8).transpose(1,2,0)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     heatimg = Image.fromarray(heatmap)
#     heatimg = heatimg.resize((256,256))
#     plt.axis('off')
#     plt.imshow(heatimg)
#     plt.savefig('{}/heatmap{}_GT{}_AS{}.png'.format(heatmap_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
#     save_image(make_grid(invTrans(outputs)), "{}/{}_GT{}_AS{}.png".format(reconstructed_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
#     save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(anomaly_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
#     orgimg= Image.open("{}/{}_GT{}_AS{}.png".format(reconstructed_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
#     plt.imshow(orgimg)
#     plt.imshow(heatimg, alpha=0.3)
#     plt.savefig('{}/{}_GT{}_AS{}.png'.format(combination_path,idx, labels_list[idx], anomaly_score_total_list[idx]))



    if anomaly_score_total_list[idx] > threshold_1 and labels_list[idx] == 0:
        misjudge_GT0_AS1 += 1
        #heatmap
        # heatmap = torch.sum(loss_func_mse(outputs, imgs), dim=1)
        # max_value = torch.max(heatmap)
        # min_value = torch.min(heatmap)
        # heatmap = (heatmap - min_value)/(heatmap - max_value)*255
        # heatmap = heatmap.detach().cpu().numpy()
        # heatmap = heatmap.astype(np.uint8).transpose(1,2,0)
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # heatimg = Image.fromarray(heatmap)
        # heatimg = heatimg.resize((256,256))
        # plt.axis('off')
        # plt.imshow(heatimg)
        # plt.savefig('{}/heatmap{}_GT{}_AS{}.png'.format(heatmap_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # save_image(make_grid(invTrans(outputs)), "{}/{}_GT{}_AS{}.png".format(reconstructed_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # orgimg= Image.open("{}/{}_GT{}_AS{}.png".format(reconstructed_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # plt.imshow(orgimg)
        # plt.imshow(heatimg, alpha=0.3)
        # plt.savefig('{}/{}_GT{}_AS{}.png'.format(combination_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    elif anomaly_score_total_list[idx] <= threshold_1 and labels_list[idx] == 1:
        misjudge_GT1_AS0 += 1
        #heatmap
        # heatmap = torch.sum(loss_func_mse(outputs, imgs), dim=1)
        # max_value = torch.max(heatmap)
        # min_value = torch.min(heatmap)
        # heatmap = (heatmap - min_value)/(heatmap - max_value)*255
        # heatmap = heatmap.detach().cpu().numpy()
        # heatmap = heatmap.astype(np.uint8).transpose(1,2,0)
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # heatimg = Image.fromarray(heatmap)
        # heatimg = heatimg.resize((256,256))
        # plt.axis('off')
        # plt.imshow(heatimg)
        # plt.savefig('{}/heatmap{}_GT{}_AS{}.png'.format(heatmap_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # save_image(make_grid(invTrans(outputs)), "{}/{}_GT{}_AS{}.png".format(reconstructed_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # orgimg= Image.open("{}/{}_GT{}_AS{}.png".format(reconstructed_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # plt.imshow(orgimg)
        # plt.imshow(heatimg, alpha=0.3)
        # plt.savefig('{}/{}_GT{}_AS{}.png'.format(combination_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    elif anomaly_score_total_list[idx] > threshold_1 and labels_list[idx] == 1:
        correct_GT1_AS1 += 1
        #heatmap
        # heatmap = torch.sum(loss_func_mse(outputs, imgs), dim=1)
        # max_value = torch.max(heatmap)
        # min_value = torch.min(heatmap)
        # heatmap = (heatmap - min_value)/(heatmap - max_value)*255
        # heatmap = heatmap.detach().cpu().numpy()
        # heatmap = heatmap.astype(np.uint8).transpose(1,2,0)
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # heatimg = Image.fromarray(heatmap)
        # heatimg = heatimg.resize((256,256))
        # plt.axis('off')
        # plt.imshow(heatimg)
        # plt.savefig('{}/heatmap{}_GT{}_AS{}.png'.format(heatmap_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # save_image(make_grid(invTrans(outputs)), "{}/{}_GT{}_AS{}.png".format(reconstructed_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # orgimg= Image.open("{}/{}_GT{}_AS{}.png".format(reconstructed_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # plt.imshow(orgimg)
        # plt.imshow(heatimg, alpha=0.3)
        # plt.savefig('{}/{}_GT{}_AS{}.png'.format(combination_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    else:
        correct_GT0_AS0 +=1
        #heatmap
        # heatmap = torch.sum(loss_func_mse(outputs, imgs), dim=1)
        # max_value = torch.max(heatmap)
        # min_value = torch.min(heatmap)
        # heatmap = (heatmap - min_value)/(heatmap - max_value)*255
        # heatmap = heatmap.detach().cpu().numpy()
        # heatmap = heatmap.astype(np.uint8).transpose(1,2,0)
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # heatimg = Image.fromarray(heatmap)
        # heatimg = heatimg.resize((256,256))
        # plt.axis('off')
        # plt.imshow(heatimg)
        # plt.savefig('{}/heatmap{}_GT{}_AS{}.png'.format(heatmap_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # save_image(make_grid(invTrans(outputs)), "{}/{}_GT{}_AS{}.png".format(reconstructed_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # orgimg= Image.open("{}/{}_GT{}_AS{}.png".format(reconstructed_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # plt.imshow(orgimg)
        # plt.imshow(heatimg, alpha=0.3)
        # plt.savefig('{}/{}_GT{}_AS{}.png'.format(combination_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
print('------------------------------------------------')
print(f'threshold_1: {threshold_1}')
print(f'misjudge_GT0_AS1: {misjudge_GT0_AS1}')
print(f'misjudge_GT1_AS0: {misjudge_GT1_AS0}')
print(f'correct_GT1_AS1: {correct_GT1_AS1}')
print(f'correct_GT0_AS0: {correct_GT0_AS0}')


misjudge_GT0_AS1 = 0
misjudge_GT1_AS0 = 0
correct_GT1_AS1 = 0
correct_GT0_AS0 = 0
for idx, (imgs) in enumerate(tqdm(test_batch)):

    if anomaly_score_total_list[idx] > threshold_2 and labels_list[idx] == 0:
        misjudge_GT0_AS1 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    elif anomaly_score_total_list[idx] <= threshold_2 and labels_list[idx] == 1:
        misjudge_GT1_AS0 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    elif anomaly_score_total_list[idx] > threshold_2 and labels_list[idx] == 1:
        correct_GT1_AS1 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    else:
        correct_GT0_AS0 +=1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
print('------------------------------------------------')
print(f'threshold_2: {threshold_2}')
print(f'misjudge_GT0_AS1: {misjudge_GT0_AS1}')
print(f'misjudge_GT1_AS0: {misjudge_GT1_AS0}')
print(f'correct_GT1_AS1: {correct_GT1_AS1}')
print(f'correct_GT0_AS0: {correct_GT0_AS0}')

misjudge_GT0_AS1 = 0
misjudge_GT1_AS0 = 0
correct_GT1_AS1 = 0
correct_GT0_AS0 = 0
for idx, (imgs) in enumerate(tqdm(test_batch)):

    if anomaly_score_total_list[idx] > threshold_3 and labels_list[idx] == 0:
        misjudge_GT0_AS1 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    elif anomaly_score_total_list[idx] <= threshold_3 and labels_list[idx] == 1:
        misjudge_GT1_AS0 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    elif anomaly_score_total_list[idx] > threshold_3 and labels_list[idx] == 1:
        correct_GT1_AS1 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    else:
        correct_GT0_AS0 +=1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))

print('------------------------------------------------')
print(f'threshold_3: {threshold_3}')
print(f'misjudge_GT0_AS1: {misjudge_GT0_AS1}')
print(f'misjudge_GT1_AS0: {misjudge_GT1_AS0}')
print(f'correct_GT1_AS1: {correct_GT1_AS1}')
print(f'correct_GT0_AS0: {correct_GT0_AS0}')

misjudge_GT0_AS1 = 0
misjudge_GT1_AS0 = 0
correct_GT1_AS1 = 0
correct_GT0_AS0 = 0
for idx, (imgs) in enumerate(tqdm(test_batch)):

    if anomaly_score_total_list[idx] > threshold_4 and labels_list[idx] == 0:
        misjudge_GT0_AS1 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    elif anomaly_score_total_list[idx] <= threshold_4 and labels_list[idx] == 1:
        misjudge_GT1_AS0 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    elif anomaly_score_total_list[idx] > threshold_4 and labels_list[idx] == 1:
        correct_GT1_AS1 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    else:
        correct_GT0_AS0 +=1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))

print('------------------------------------------------')
print(f'threshold_4: {threshold_4}')
print(f'misjudge_GT0_AS1: {misjudge_GT0_AS1}')
print(f'misjudge_GT1_AS0: {misjudge_GT1_AS0}')
print(f'correct_GT1_AS1: {correct_GT1_AS1}')
print(f'correct_GT0_AS0: {correct_GT0_AS0}')

misjudge_GT0_AS1 = 0
misjudge_GT1_AS0 = 0
correct_GT1_AS1 = 0
correct_GT0_AS0 = 0
for idx, (imgs) in enumerate(tqdm(test_batch)):

    if anomaly_score_total_list[idx] > threshold_5 and labels_list[idx] == 0:
        misjudge_GT0_AS1 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    elif anomaly_score_total_list[idx] <= threshold_5 and labels_list[idx] == 1:
        misjudge_GT1_AS0 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    elif anomaly_score_total_list[idx] > threshold_5 and labels_list[idx] == 1:
        correct_GT1_AS1 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    else:
        correct_GT0_AS0 +=1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))

print('------------------------------------------------')
print(f'threshold_5: {threshold_5}')
print(f'misjudge_GT0_AS1: {misjudge_GT0_AS1}')
print(f'misjudge_GT1_AS0: {misjudge_GT1_AS0}')
print(f'correct_GT1_AS1: {correct_GT1_AS1}')
print(f'correct_GT0_AS0: {correct_GT0_AS0}')

misjudge_GT0_AS1 = 0
misjudge_GT1_AS0 = 0
correct_GT1_AS1 = 0
correct_GT0_AS0 = 0
for idx, (imgs) in enumerate(tqdm(test_batch)):

    if anomaly_score_total_list[idx] > threshold_6 and labels_list[idx] == 0:
        misjudge_GT0_AS1 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    elif anomaly_score_total_list[idx] <= threshold_6 and labels_list[idx] == 1:
        misjudge_GT1_AS0 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    elif anomaly_score_total_list[idx] > threshold_6 and labels_list[idx] == 1:
        correct_GT1_AS1 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    else:
        correct_GT0_AS0 +=1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))

print('------------------------------------------------')
print(f'threshold_6: {threshold_6}')
print(f'misjudge_GT0_AS1: {misjudge_GT0_AS1}')
print(f'misjudge_GT1_AS0: {misjudge_GT1_AS0}')
print(f'correct_GT1_AS1: {correct_GT1_AS1}')
print(f'correct_GT0_AS0: {correct_GT0_AS0}')

misjudge_GT0_AS1 = 0
misjudge_GT1_AS0 = 0
correct_GT1_AS1 = 0
correct_GT0_AS0 = 0
for idx, (imgs) in enumerate(tqdm(test_batch)):

    if anomaly_score_total_list[idx] > threshold_7 and labels_list[idx] == 0:
        misjudge_GT0_AS1 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    elif anomaly_score_total_list[idx] <= threshold_7 and labels_list[idx] == 1:
        misjudge_GT1_AS0 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(misjudge_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    elif anomaly_score_total_list[idx] > threshold_7 and labels_list[idx] == 1:
        correct_GT1_AS1 += 1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))
    else:
        correct_GT0_AS0 +=1
        # save_image(make_grid(invTrans(torch.cat([imgs, outputs]))), "{}/{}_GT{}_AS{}.png".format(correct_figure_path,idx, labels_list[idx], anomaly_score_total_list[idx]))

print('------------------------------------------------')
print(f'threshold_7: {threshold_7}')
print(f'misjudge_GT0_AS1: {misjudge_GT0_AS1}')
print(f'misjudge_GT1_AS0: {misjudge_GT1_AS0}')
print(f'correct_GT1_AS1: {correct_GT1_AS1}')
print(f'correct_GT0_AS0: {correct_GT0_AS0}')