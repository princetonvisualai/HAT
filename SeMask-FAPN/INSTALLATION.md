We give a detail instruction of the installation of SeMask-FaPN. The instruction is adopted from the original repository, [SeMask-FaPN](https://github.com/Picsart-AI-Research/SeMask-Segmentation/tree/main/SeMask-FAPN). Please refer to the [issues](https://github.com/Picsart-AI-Research/SeMask-Segmentation/issues) of the original repository on any installation issues.

### Setting Up PyTorch

This tool requires PyTorch [1.7.1](https://pytorch.org/get-started/previous-versions/#v171). 
If this version of the PyTorch does not work with your intended human action recognition model, you can use [Anaconda](https://anaconda.org) environments. 

```
conda create --name SeMask python=3

# after finishing creating the environment

conda activate SeMask
```

You can then install [PyTorch 1.7.1](https://pytorch.org/get-started/previous-versions/#v171) following the website instructions.

```
# E.g.,
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```


### Build DCNv2

[Original instructions](https://github.com/Picsart-AI-Research/SeMask-Segmentation/tree/main/SeMask-FAPN/DCNv2)

Note that you need to have GCC version above 5.0.

```
# HAT/SeMask-FAPN/DCNv2

sh ./make.sh         # build
python test/test.py    # run examples and gradient check 
```


### Detectron2 v0.5

Install Detectron2 v0.5 that is compatible to your current setup.
[https://github.com/facebookresearch/detectron2/releases](https://github.com/facebookresearch/detectron2/releases)

```
e.g.)
python -m pip install detectron2==0.5 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
```

### Install Mask2Former requirements and CUDA kernel for MSDeformAttn

```
# HAT/SeMask-FAPN/SeMask-Mask2Former
pip install -r requirements.txt

# After finishing the installation

cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

## Download Weights

Go to [https://github.com/Picsart-AI-Research/SeMask-Segmentation/tree/main/SeMask-FAPN](https://github.com/Picsart-AI-Research/SeMask-Segmentation/tree/main/SeMask-FAPN) and download the weights for **SeMask-L Mask2Former MSFaPN**. Put the file under this diretory.

```
i.e., 
HAT/SeMask-FAPN/semask_large_mask2former_msfapn_ade20k.pth
```

## Testing the Installation.

```
# HAT/SeMask-FAPN/SeMask-Mask2Former
python demo/demo.py
```

See if `demo/output.jpg` is a human segmentation of `demo/input.jpg`.
