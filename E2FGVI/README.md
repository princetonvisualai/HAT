# E<sup>2</sup>FGVI (CVPR 2022)

The code is based on [E<sup>2</sup>FGVI](https://github.com/MCG-NKU/E2FGVI). The code is slightly modified to fit our use. 

## Environment Setup

This tool requires PyTorch >= 1.5. If you have already installed PyTorch 1.7.1 from SeMask, you can keep using that environment.

## Dependencies

Install mmcv-full from the official [website](https://github.com/open-mmlab/mmcv#installation)

```
# e.g.,
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
```

Install other dependencies:
```
conda install scikit-image==0.16.2
pip install tqdm 
```

## Download pretrained model

Download the pretrained model from the [original repository](https://github.com/MCG-NKU/E2FGVI#prepare-pretrained-models), and save it under `release_model`.


I.e.,
```
release_model
   |- E2FGVI-CVPR22.pth
```

## Generating Inpainting

```
# HAT/E2FGVI
python generate.py --model e2fgvi --video "../data/kinetics400/ori/*/*" --ckpt release_model/E2FGVI-CVPR22.pth
```

If this causes a CUDA memory error, use higher steps. Using higher steps might reduce the quality of the inpainted frames.

```
# HAT/E2FGVI
python generate.py --model e2fgvi --video "../data/kinetics400/ori/*/*" --ckpt release_model/E2FGVI-CVPR22.pth --step=20
```