# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
from tqdm import tqdm
import torch
import glob

from core.utils import to_tensors

parser = argparse.ArgumentParser(description="E2FGVI")
parser.add_argument("-v", "--video", type=str, required=True)
parser.add_argument("-c", "--ckpt", type=str, required=True)
parser.add_argument("--model", type=str, choices=['e2fgvi', 'e2fgvi_hq'])
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=5)
parser.add_argument("--savefps", type=int, default=24)

# args for e2fgvi_hq (which can handle videos with arbitrary resolution)
parser.add_argument("--set_size", action='store_true', default=False)
parser.add_argument("--width", type=int)
parser.add_argument("--height", type=int)

args = parser.parse_args()

ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps


# sample reference frames from the whole video
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length-1, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index



# read frame-wise masks
def read_mask(mpath, size):
    masks = []
    mnames = sorted(glob.glob(os.path.join(mpath, '*.png')))
    
    for mp in mnames:
        m = Image.open(mp)
        m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 128).astype(np.uint8)
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                       iterations=4)
        masks.append(Image.fromarray(m * 255))
    return masks



def read_frame(vname):
    
    frames = []
    # lst = os.listdir(vname)
    # lst.sort()
    # fr_lst = [vname + '/' + name for name in lst]
    fr_lst = sorted(glob.glob(os.path.join(vname, '*.jpg')))
    for fr in fr_lst:
        image = cv2.imread(fr)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image)
    return frames, fr_lst, image.size
    


# resize frames
def resize_frames(frames, size=None):
    if size is not None:
        frames = [f.resize(size) for f in frames]
    else:
        size = frames[0].size
    return frames, size


# set up models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model == "e2fgvi":
    set_size = (432, 240)
elif args.set_size:
    set_size = (args.width, args.height)
else:
    set_size = None

net = importlib.import_module('model.' + args.model)
model = net.InpaintGenerator().to(device)
data = torch.load(args.ckpt, map_location=device)
model.load_state_dict(data)
print(f'Loading model from: {args.ckpt}')
model.eval()


import glob

jihoon_videos = []
for v in sorted(glob.glob(args.video)):
    jihoon_videos.append([v, v.replace('/ori/','/seg/')])
    
for jihoon_video in tqdm(jihoon_videos):
    
    frames, frames_names, ori_size = read_frame(jihoon_video[0])
    frames, frame_size = resize_frames(frames, set_size)
    h, w = frame_size[1], frame_size[0]
    video_length = len(frames)
    imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
    frames = [np.array(f).astype(np.uint8) for f in frames]

    masks = read_mask(jihoon_video[1], frame_size)
    binary_masks = [
        np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks
    ]
    masks = to_tensors()(masks).unsqueeze(0)
    # imgs, masks = imgs.to(device), masks.to(device)
    comp_frames = [None] * video_length
    
    
    for f in (range(0, video_length, neighbor_stride)):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                             min(video_length, f + neighbor_stride + 1))
        ]
        # print(neighbor_ids)
        ref_ids = get_ref_index(f, neighbor_ids, video_length)
        selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
        with torch.no_grad():
            masked_imgs = selected_imgs * (1 - selected_masks)
            mod_size_h = 60
            mod_size_w = 108
            h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
            w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [3])],
                3)[:, :, :, :h + h_pad, :]
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [4])],
                4)[:, :, :, :, :w + w_pad]
            pred_imgs, _ = model(masked_imgs.cuda(), len(neighbor_ids))
            pred_imgs = pred_imgs[:, :, :h, :w]
            pred_imgs = (pred_imgs + 1) / 2
            pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_imgs[i]).astype(
                    np.uint8) * binary_masks[idx] + frames[idx] * (
                        1 - binary_masks[idx])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32) * 0.5 + img.astype(np.float32) * 0.5
    
    for f in range(video_length):
        name = frames_names[f].replace('/ori/','/inp/')
        os.makedirs(os.path.dirname(name), exist_ok=True)
        comp = comp_frames[f].astype(np.uint8)
        cv2.imwrite(name, cv2.cvtColor(cv2.resize(comp, ori_size), cv2.COLOR_BGR2RGB))