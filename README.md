# Multispectral conditional Generative Adversarial Nets
This repository is an implementation of ["Filmy Cloud Removal on Satellite Imagery with Multispectral Conditional Generative Adversarial Nets"](https://arxiv.org/abs/1710.04835).

## Requirements
* Python
* PyTorch
* TorchVision
* Numpy
* OpenCV
* Pillow
* tqdm
* PyYAML

## Preparing training data
Please refer to [make_dataset/README.md](make_dataset/README.md).

## How to train
You need set each parameters in the `config.yml`.  
`config.yml` is automatically copied to `out_dir`.  

```bash
python train.py
```

## How to test

```bash
python predict.py --config <path to config.yml in the out_dir> --test_dir <path to a directory stored test data> --out_dir <path to an output directory> --pretrained <path to a pretrained model> --cuda
```
