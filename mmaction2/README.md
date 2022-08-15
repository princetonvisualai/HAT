# MMAction2

This directory is a smaller version of [MMAction2](https://github.com/open-mmlab/mmaction2) to showcase HAT toolkit. 
The only added features are [actionswap_dataset.py](mmaction/datasets/actionswap_dataset.py) and [humanframe_dataset.py](mmaction/datasets/humanframe_dataset.py).

Please refer to the code above to implement HAT in your own project, or to use in your own clone of MMAction2. 

## Installation

Prerequisits for MMAction2 were already installed in the previous steps. Please install MMAction2 using

```
pip install -v -e .
```

Refer to the [original repository](https://github.com/open-mmlab/mmaction2/blob/master/docs/install.md) for full installation procedure if needed.

## Use of HAT in MMAction2

### Dataset Preparation

MMAction2 requires an addition annotation file that contains the location of the video (images), length of the video, and class id. 
MMAction2 offers [codes](https://github.com/open-mmlab/mmaction2/blob/master/docs/data_preparation.md#generate-file-list) to generate the annotation file. 
See the [demo file](../data/kinetics400/kinetics400_val_list_rawframes.txt) for the format. 

### Demo

Here we demonstrate use of HAT in [TSN](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py) model. 
While you can use our demo videos, note that the demo videos are from the training set of Kinetics400. 

### Download Checkpoint

Download the trained weights [link](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) and put them under `checkpoints` folder.

I.e.,
```
HAT/mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth
```

### Original Accuracy

```
# at HAT/mmaction2
python tools/test.py ./configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
./checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
--out ori.json \
--cfg-options data.workers_per_gpu=16 \
    data.test_dataloader.videos_per_gpu=1 \
    data.test.filename_tmpl={:06}.jpg \
    data.test.ann_file="../data/kinetics400/kinetics400_val_list_rawframes.txt" \
    data.test.data_prefix="../data/kinetics400/ori" \
    data.test.start_index=1 \
--eval top_k_accuracy mean_class_accuracy
```

### Background Only Accuracy

```
# at HAT/mmaction2
python tools/test.py ./configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
./checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
--out inp.json \
--cfg-options data.workers_per_gpu=16 \
    data.test_dataloader.videos_per_gpu=1 \
    data.test.filename_tmpl={:06}.jpg \
    data.test.ann_file="../data/kinetics400/kinetics400_val_list_rawframes.txt" \
    data.test.data_prefix="../data/kinetics400/inp" \
    data.test.start_index=1 \
--eval top_k_accuracy mean_class_accuracy
```

### Human Only Accuracy

```
python tools/test.py ./configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
./checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
--out humanonly.json \
--cfg-options \
    data.workers_per_gpu=16 \
    data.test.type=HumanframeDataset \
    data.test.rgb_data_prefix="../data/kinetics400/ori/" \
    data.test.seg_data_prefix="../data/kinetics400/seg/" \
    data.test.rgb_filename_tmpl={:06}.jpg \
    data.test.seg_filename_tmpl={:06}.png \
    data.test.start_index=1 \
    data.test.data_prefix="" \
    data.test_dataloader.videos_per_gpu=1 \
    data.test.ann_file="../data/kinetics400/kinetics400_val_list_rawframes.txt" \
--eval top_k_accuracy mean_class_accuracy
```

### Action Swap Accuracy

Refer to [pickle files](    data) for the action swap pairs that we used for our original paper. 
The pickle file contains a python dictionary file where the key is the foreground action video and the value is a list where

`[background_video_name, background_video_frame_number, background_video_class_index]`.

We have included a pair file for demo purposes.

```
python tools/test.py ./configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
./checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
--out swap.json \
--cfg-options \
    data.test_dataloader.videos_per_gpu=1 \
    data.test.ann_file="../data/kinetics400/kinetics400_val_list_rawframes.txt" \
    data.test.type=ActionSwapDataset \
    data.test.data_prefix="" \
    data.test.swapfile=data/actionswap_demo.pickle \
    data.test.foreground_rgb_prefix='../data/kinetics400/ori/' \
    data.test.foreground_seg_prefix='../data/kinetics400/seg/' \
    data.test.background_rgb_prefix='../data/kinetics400/inp/' \
    data.test.background_seg_prefix='../data/kinetics400/seg/' \
    data.test.foreground_rgb_start_index=1 \
    data.test.foreground_seg_start_index=1 \
    data.test.background_rgb_start_index=1 \
    data.test.background_seg_start_index=1 \
    data.test.foreground_rgb_filename_tmpl='{:06}.jpg' \
    data.test.foreground_seg_filename_tmpl='{:06}.png' \
    data.test.background_rgb_filename_tmpl='{:06}.jpg' \
    data.test.background_seg_filename_tmpl='{:06}.png' \
--eval top_k_accuracy mean_class_accuracy
```

The output gives the accuracy calculated on the foreground action class labels. You can go through `swap.json` to calculate the error caused by the background. 