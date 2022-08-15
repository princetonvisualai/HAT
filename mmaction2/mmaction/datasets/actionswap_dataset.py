# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import os

import torch
from scipy import ndimage
import numpy as np
from PIL import Image

from mmaction.datasets.pipelines import Resize
from .base import BaseDataset
from .builder import DATASETS
from .pipelines import Compose

import pickle
@DATASETS.register_module()
class ActionSwapDataset(BaseDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 163 1
        some/directory-2 122 1
        some/directory-3 258 2
        some/directory-4 234 2
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a multi-class annotation file:


    .. code-block:: txt

        some/directory-1 163 1 3 5
        some/directory-2 122 1 2
        some/directory-3 258 2
        some/directory-4 234 2 4 6 8
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a with_offset annotation file (clips from long videos), each
    line indicates the directory to frames of a video, the index of the start
    frame, total frames of the video clip and the label of a video clip, which
    are split with a whitespace.


    .. code-block:: txt

        some/directory-1 12 163 3
        some/directory-2 213 122 4
        some/directory-3 100 258 5
        some/directory-4 98 234 2
        some/directory-5 0 295 3
        some/directory-6 50 121 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Default: False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int | None): Number of classes in the dataset.
            Default: None.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0.,
                 dynamic_length=False,
                 swapfile='./data/actionswap_demo.pickle',
                 foreground_rgb_prefix='../data/kinetics400/ori/',
                 foreground_seg_prefix='../data/kinetics400/seg/',
                 background_rgb_prefix='../data/kinetics400/inp/',
                 background_seg_prefix='../data/kinetics400/seg/',
                 foreground_rgb_start_index=1,
                 foreground_seg_start_index=1,
                 background_rgb_start_index=1,
                 background_seg_start_index=1,
                 foreground_rgb_filename_tmpl='{:06}.jpg',
                 foreground_seg_filename_tmpl='{:06}.png',
                 background_rgb_filename_tmpl='{:06}.jpg',
                 background_seg_filename_tmpl='{:06}.png',
                 **kwargs):
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        self.foreground_rgb_prefix = foreground_rgb_prefix
        self.foreground_seg_prefix = foreground_seg_prefix
        self.background_rgb_prefix = background_rgb_prefix
        self.background_seg_prefix = background_seg_prefix
        self.foreground_rgb_start_index = foreground_rgb_start_index
        self.foreground_seg_start_index = foreground_seg_start_index
        self.background_rgb_start_index = background_rgb_start_index
        self.background_seg_start_index = background_seg_start_index
        self.foreground_rgb_filename_tmpl = foreground_rgb_filename_tmpl
        self.foreground_seg_filename_tmpl = foreground_seg_filename_tmpl
        self.background_rgb_filename_tmpl = background_rgb_filename_tmpl
        self.background_seg_filename_tmpl = background_seg_filename_tmpl
        assert data_prefix==''
        
        pipeline_1 = []
        pipeline_2 = []
        assert pipeline[1]['type'] == 'RawFrameDecode'
        pipeline_decoder = [pipeline[1]]
        first = True
        for line in pipeline:
            if 'Resize' in line['type']:
                first = False
                
            if first:
                pipeline_1.append(line.copy())
            else:
                pipeline_2.append(line.copy())
            
                
        pipeline = pipeline_1
        self.pipeline_2 = Compose(pipeline_2)
        self.pipeline_decoder = Compose(pipeline_decoder)
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            multi_class,
            num_classes,
            start_index,
            modality,
            sample_by_class=sample_by_class,
            power=power,
            dynamic_length=dynamic_length)
        self.short_cycle_factors = kwargs.get('short_cycle_factors',
                                              [0.5, 0.7071])
        self.default_s = kwargs.get('default_s', (224, 224))
        
        
        
        
        # my code below #############################################
        self.mapping = pickle.load(open(swapfile, 'rb'))
            
        #############################################################

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                
                line_split_temp = []
                line_split_temp.append(line_split[-1])
                line_split_temp.append(line_split[-2])
                line_split_temp.append(' '.join(line_split[:-2]))
                line_split = line_split_temp[::-1]
                
                video_info = {}
                idx = 0
                # idx for frame_dir
                frame_dir = line_split[idx]
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)
                video_info['frame_dir'] = frame_dir
                idx += 1
                if self.with_offset:
                    # idx for offset and total_frames
                    video_info['offset'] = int(line_split[idx])
                    video_info['total_frames'] = int(line_split[idx + 1])
                    idx += 2
                else:
                    # idx for total_frames
                    video_info['total_frames'] = int(line_split[idx])
                    idx += 1
                # idx for label[s]
                label = [int(x) for x in line_split[idx:]]
                assert label, f'missing label in line: {line}'
                if self.multi_class:
                    assert self.num_classes is not None
                    video_info['label'] = label
                else:
                    assert len(label) == 1
                    video_info['label'] = label[0]
                video_infos.append(video_info)

        return video_infos

    def prepare_train_frames(self, idx):
        raise NotImplementedError('lol')

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        
        foreground_name = self.video_infos[idx]['frame_dir']
        results = copy.deepcopy(self.video_infos[idx])
        results['frame_dir'] = os.path.join(self.foreground_rgb_prefix,results['frame_dir'])
        results['filename_tmpl'] = self.foreground_rgb_filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.foreground_rgb_start_index
        imgs = self.pipeline(results)
        
        
        
        # foreground seg
        results = copy.deepcopy(self.video_infos[idx])
        results['frame_dir'] = os.path.join(self.foreground_seg_prefix,results['frame_dir'])
        results['filename_tmpl'] = self.foreground_seg_filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.foreground_seg_start_index
        segs = self.pipeline(results)
        
        # now we need to load background videos and segmentations.
        # frame sampling is done by taking the same relative position, based on the length of the video
        # resizing is done by setting the short-side to be the same as foreground short-side.
        # this fixed-rule will work same as "pre-generated" video, but generated in real-time, and can work and any arbituary temporal sampling / resizing
        
        # first, lets get frame index for the background
        
        background_idx = ((imgs['frame_inds'] - imgs['start_index']) / imgs['total_frames'] * self.mapping[foreground_name][1]).round().astype(int).clip(0, self.mapping[foreground_name][1]-1) + self.background_rgb_start_index
        background_seg_idx = ((imgs['frame_inds'] - imgs['start_index']) / imgs['total_frames'] * self.mapping[foreground_name][1]).round().astype(int).clip(0, self.mapping[foreground_name][1]-1) + self.background_seg_start_index
        
        if imgs['imgs'][0].shape[0] > imgs['imgs'][0].shape[1]:
            resize = Resize(scale=(-1, imgs['imgs'][0].shape[1]))
        else:
            resize = Resize(scale=(imgs['imgs'][0].shape[0], -1))
            
        background = {
            'frame_dir': os.path.join(self.background_rgb_prefix,self.mapping[foreground_name][0]),
            'total_frames': self.mapping[foreground_name][1],
            'label': self.mapping[foreground_name][2],
            'filename_tmpl': self.background_rgb_filename_tmpl,
            'modality': self.modality,
            'start_index':self.background_rgb_start_index,
            'frame_inds': background_idx
        }
        
        background_seg = {
            'frame_dir': os.path.join(self.foreground_seg_prefix,self.mapping[foreground_name][0]),
            'total_frames': self.mapping[foreground_name][1],
            'label': self.mapping[foreground_name][2],
            'filename_tmpl': self.background_seg_filename_tmpl,
            'modality': self.modality,
            'start_index':self.background_seg_start_index,
            'frame_inds': background_seg_idx
        }
        if 'num_clips' in imgs.keys():
            background_seg['num_clips'] = imgs['num_clips']
            background['num_clips'] = imgs['num_clips']
            background_seg['frame_interval'] = imgs['frame_interval']
            background['frame_interval'] = imgs['frame_interval']
            background_seg['clip_len'] = imgs['clip_len']
            background['clip_len'] = imgs['clip_len']
            
        background = resize(self.pipeline_decoder(background))
        background_seg = resize(self.pipeline_decoder(background_seg))
            
        
        
        
        
        # merge forground and background
        img_list = []
        
        if segs['imgs'][0].sum() > 0:
            forground_center  = ndimage.measurements.center_of_mass(segs['imgs'][0])[:2]   # center on width
        else:
            forground_center = (segs['imgs'][0].shape[0] / 2,segs['imgs'][0].shape[1] / 2)
            
        if background_seg['imgs'][0].sum() > 0:
            background_center = ndimage.measurements.center_of_mass(background_seg['imgs'][0])[:2]   # center on width
        else:
            background_center = (background_seg['imgs'][0].shape[0]/2, background_seg['imgs'][0].shape[1]/2)
            
        # imgs['imgs'][0][int(forground_center[0]), int(forground_center[1])] = [255, 0, 0]
            
        move = (int(background_center[0] - forground_center[0]), int(background_center[1] - forground_center[1]))
        
        background_imgs = []
        final_imgs = []
        for i, img in enumerate(imgs['imgs']):
            
            seg = (segs['imgs'][i] > 128)[:,:,0]
            backg = background['imgs'][i]
            background_imgs.append(backg.copy())
            
            backg_t  = np.zeros_like(backg)
            
            img = Image.fromarray(img)
            backg = Image.fromarray(backg)
            backg.paste(img, (move[1], move[0]), Image.fromarray(seg).convert('L'))
            
            
            img_list.append(np.array(backg))
            final_imgs.append(np.array(backg))
            
        background['imgs'] = img_list
        final = self.pipeline_2(background)
        
        output = {}
        output['imgs'] = final['imgs']
        output['label'] = final['label']
        # output['vidid'] = self.video_infos[idx]['frame_dir']
        # output['backid'] = self.video_infos[self.mapping[idx]]['frame_dir']
        
        # output = {}
        # output['forground'] = imgs['imgs']
        # output['forground_seg'] = segs['imgs']
        # output['background'] = background_imgs
        # output['background_seg'] = background_seg['imgs']
        # output['final'] = final_imgs
        return output