# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config


# constants
WINDOW_NAME = "mask2former demo"


import glob
from PIL import Image
import numpy as np
import os
import torch

import argparse

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./configs/msfapn_maskformer2_semask_swin_large_IN21k_384_bs16_160k_res640.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        default='../../data/kinetics400'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

    
args = get_parser().parse_args()
cfg = setup_cfg(args)


import detectron2.data.transforms as T
import math
class dataset(torch.utils.data.Dataset):
    
    def __init__(self, imgs, batchsize):

        self.imgs = imgs
        
        # This is used, assuming that images in the same folder (thus the same video), have same image size
        # and thus can be batched. 
        self.batch_sampler = []
        for i, img in enumerate(imgs):
            
            if len(self.batch_sampler) == 0 or len(self.batch_sampler[-1]) >= batchsize or os.path.dirname(imgs[self.batch_sampler[-1][-1]]) != os.path.dirname(img):
                self.batch_sampler.append([])
                
            self.batch_sampler[-1].append(i)
            
            
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = read_image(self.imgs[idx], format="RGB")
        height, width = img.shape[:2]
        
        img = self.aug.get_transform(img).apply_image(img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        
        return img, height, width, self.imgs[idx]
    
    
    

imgs = sorted(glob.glob(os.path.join(args.input, 'ori', '**','*.jpg'), recursive=True))
print('num of images ', len(imgs))


d = dataset(imgs=imgs, batchsize=4)
l = torch.utils.data.DataLoader(d, num_workers=16, batch_sampler=d.batch_sampler)

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
model = build_model(cfg)  # returns a torch.nn.Module
model.training = False
model.eval()

checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)

with torch.no_grad():
    for img, height, width, name in tqdm.tqdm(l):

        batchsize = img.shape[0]
        inputs = []
        for i in range(batchsize):
            inputs.append({'image': img[i], 'height': height[i], 'width': width[i]})

        output = model(inputs)

        for o, n in zip(output, name):
            person = ((o['sem_seg'][12].detach().cpu().numpy() * 255).clip(0,255)).astype(np.uint8)
            person = Image.fromarray(person)

            path_output = n.replace(os.path.join(args.input, 'ori'), os.path.join(args.input, 'seg'))[:-4] + ".png"
            os.makedirs(os.path.dirname(path_output), exist_ok=True)
            person.save(path_output)
        
        del output