The code is based on the official release code of [SeMask-FaPN](https://github.com/Picsart-AI-Research/SeMask-Segmentation/tree/main/SeMask-FAPN). The code is slightly modified to generate only the human segmentations. The generated human segmatations will be saved in `data/kinetics/seg`.

## Installation

Please follow the [instructions](INSTALLATION.md).

## Generate Segmentations

```
# /HAT/SeMask-FAPN/SeMask-Mask2Former
python demo/generate.py --input ../../data/kinetics400 --opts MODEL.WEIGHTS ../semask_large_mask2former_msfapn_ade20k.pth 
```