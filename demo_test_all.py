import argparse
import json
import os

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
from config import *
from model import SCNN
from utils.prob2lines import getLane
from utils.transforms import *

import cv2
import matplotlib.pyplot as plt
#%matplotlib inline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/exp0")
    parser.add_argument("--dataset_split", type=str, default="test")
    args = parser.parse_args()
    return args


# ------------ config ------------
args = parse_args()
exp_dir = args.exp_dir
exp_name = exp_dir.split('/')[-1]

dataset_split = args.dataset_split

#exp_dir = "./experiments/exp0"
#exp_name = exp_dir.split('/')[-1]
#dataset_split = "test"

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
resize_shape = tuple(exp_cfg['dataset']['resize_shape'])
device = torch.device('cuda')


# ------------ data and model ------------
dataset_name = exp_cfg['dataset']['dataset_name']
if dataset_name == 'Tusimple':
    # Imagenet mean, std
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    original_shape = (1280,720)
    nb_img = 20
elif dataset_name == 'CULane':
    # CULane mean, std
    mean=(0.3598, 0.3653, 0.3662)
    std=(0.2573, 0.2663, 0.2756)
    original_shape = (1640, 590)
    nb_img = 1

    
    
transform = Compose(Resize(resize_shape), ToTensor(),
                    Normalize(mean=mean, std=std))
Dataset_Type = getattr(dataset, dataset_name)

test_dataset = Dataset_Type(Dataset_Path[dataset_name], dataset_split, transform)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.collate, num_workers=4)

net = SCNN(input_size=resize_shape, pretrained=False)
save_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '_best.pth')
save_dict = torch.load(save_name, map_location='cpu')

#print("\nloading", save_name, "...... From Epoch: ", save_dict['epoch'])
net.load_state_dict(save_dict['net'])
net = torch.nn.DataParallel(net.to(device))
net.eval()

progressbar = tqdm(range(len(test_loader)))

for batch_idx, sample in enumerate(test_loader):
    #import pdb; pdb.set_trace()
    for i_img in range(nb_img):
        if dataset_name == 'Tusimple':
            img_directory = sample['img_name'][0][0:-6]
            img_name = img_directory + str(i_img+1) + ".jpg"
        elif dataset_name == 'CULane':
            img_name = sample['img_name'][0]
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        sample2 = {}
        sample2['img'] = img
        sample2 = transform(sample2)
        sample['img'] = sample2['img']
        
        img = sample['img']
        img = img.unsqueeze(0).to(device)
        #img_name = sample['img_name']

        with torch.no_grad():
            seg_pred, exist_pred = net(img)[:2]
        seg_pred = F.softmax(seg_pred, dim=1)
        seg_pred = seg_pred.cpu().numpy()
        exist_pred = exist_pred.cpu().numpy()

        b=0
        seg = seg_pred[b]
        exist = [1 if exist_pred[b, i] > 0.5 else 0 for i in range(4)]
        if dataset_name == 'Tusimple':
            lane_coords = getLane.prob2lines_tusimple(seg, exist, resize_shape=original_shape[::-1], y_px_gap=10, pts=56)
        elif dataset_name == 'CULane':
            lane_coords = getLane.prob2lines_CULane(seg, exist, resize_shape=original_shape[::-1], y_px_gap=20, pts=18)
        for i in range(len(lane_coords)):
            lane_coords[i] = sorted(lane_coords[i], key=lambda pair: pair[1])
            
        img_vis = cv2.imread(img_name)
        #img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
        for lane in lane_coords:
            foo = np.int32([lane])[:,np.int32([lane])[0,:,0] >= 0]
            cv2.polylines(img_vis, foo, isClosed=False, color=(0,0,255), thickness=2)

        #show segmentation map
        seg = seg.swapaxes(0,2).swapaxes(0,1)
        seg = cv2.resize(seg, dsize=original_shape, interpolation=cv2.INTER_CUBIC)

        lane_img = np.zeros_like(img_vis)
        color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
        coord_mask = np.argmax(seg, axis=-1)
        for i in range(0, 4):
            if exist_pred[0, i] > 0.5:
                lane_img[coord_mask == (i + 1)] = color[i]
                    
        img_vis = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img_vis, beta=1., gamma=0.)
        
        cv2.imshow("da window", img_vis)
        cv2.waitKey(1)



