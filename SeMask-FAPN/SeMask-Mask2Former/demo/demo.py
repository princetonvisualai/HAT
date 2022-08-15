# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import tempfile
import time
import warnings

import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo

# constants
WINDOW_NAME = "mask2former demo"

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
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

        
        
args = get_parser().parse_args('--config-file ./configs/msfapn_maskformer2_semask_swin_large_IN21k_384_bs16_160k_res640.yaml --input input.jpg   --output output.jpg   --opts MODEL.WEIGHTS ../semask_large_mask2former_msfapn_ade20k.pth'.split())
cfg = setup_cfg(args)
demo = VisualizationDemo(cfg)



import glob
from PIL import Image
import numpy as np
import os

img = read_image('./demo/input.jpg', format="BGR")
predictions, visualized_output = demo.run_on_image(img)
person = ((predictions['sem_seg'][12].detach().cpu().numpy() * 255).clip(0,255)).astype(np.uint8)
person = Image.fromarray(person)
person.save('./demo/output.jpg')