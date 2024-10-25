import cv2
import os
import torch
import argparse
import numpy as np
from PIL import Image
from model.Reconstruction import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from model.Reconstruction import *

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=15, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
parser.add_argument('--model_dir', type=str, help='directory of model')
parser.add_argument('--m_items_dir', type=str, help='directory of model')
parser.add_argument('--exp_dir', type=str, default='evaluate_log', help='directory of log')
parser.add_argument('--figure_dir', type=str, default='./figures/', help='directory of log')
parser.add_argument('--heatmap_dir', type=str, default='./heatmap', help='directory of log')
parser.add_argument('--img_path', type=str, default='./dataset/ped2/testing/frames/01/000.jpg', help='directory of log')
args = parser.parse_args()

if not os.path.exists(args.heatmap_dir):
    os.makedirs(args.heatmap_dir)

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
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

img = Image.open(args.img_path)
input_tensor = transform(img)
input_batch = input_tensor.unsqueeze(0)

#load model
model = torch.load(args.model_dir)
model.cuda()
m_items = torch.load(args.m_items_dir)
m_items_test = m_items.clone()
model.eval()

# with torch.no_grid():
# 	outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(img, m_items_test, False)

outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(img, m_items_test, False)
heatmap = torch.sum(outputs, dim=1)
max_value = torch.max(heatmap)
min_value = torch.min(heatmap)
heatmap = (heatmap - min_value)/(heatmap - max_value)*255

heatmap = heatmap.cpu().numpy().astype(np.unit8).transpose(1,2,0)

heatmap = cv2.applyColormap(heatmap, cv2.COLORMAP_JET)
heatimg = Image.fromarray(heatmap)

org_size = img.size
heatimg = heatimg.resize(org_size)
plt.savefig('./heatmap/000.jpg')