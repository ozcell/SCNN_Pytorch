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
    parser.add_argument("--video_path", type=str, default="~/dataset/fordata/V1.mp4")
    args = parser.parse_args()
    return args


# ------------ config ------------
args = parse_args()
exp_dir = args.exp_dir
exp_name = exp_dir.split('/')[-1]

video_path = args.video_path

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
elif dataset_name == 'CULane':
    # CULane mean, std
    mean=(0.3598, 0.3653, 0.3662)
    std=(0.2573, 0.2663, 0.2756)
original_shape = (1280,720)

    
    
transform = Compose(Resize(resize_shape), ToTensor(),
                    Normalize(mean=mean, std=std))


cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Video file cannot be opened"

net = SCNN(input_size=resize_shape, pretrained=False)
save_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '_best.pth')
save_dict = torch.load(save_name, map_location='cpu')

#print("\nloading", save_name, "...... From Epoch: ", save_dict['epoch'])
net.load_state_dict(save_dict['net'])
net = torch.nn.DataParallel(net.to(device))
net.eval()

#progressbar = tqdm(range(len(test_loader)))
frame_count = 0
while cap.isOpened():
    ret, img = cap.read()
    if frame_count == 15:
        frame_count = 0
    else:
        frame_count += 1
        continue
    if ret:
        img = cv2.resize(img, dsize=original_shape, interpolation=cv2.INTER_CUBIC)
        img_vis = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform({'img': img})['img']
        img = img.unsqueeze(0).to(device)

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
    
    else:
        cap.release()

cv2.destroyAllWindows()



