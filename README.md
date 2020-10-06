# MAIC2020_challenge

This repository is codebase for `MAIC2020_challenge`(http://maic.or.kr/competitions/1/infomation) developed by the AI team at KNUH(Kyongbook Nation Univ. Hospital). The aim of this challenge is to predict **hypotension before 5 minutes by utilizing the 20 seconds radial arterial pressure waveform** measured during surgery.

This codebase is made to help other researchers and industry practitioners:

- reproduce some of our research results and
- leverage our very strong pre-trained models.

Currently, we support following models:

- 1D-ResNet (basic)
- 1D-ResNet w/ **non-local**

  - For more details, refer to [link](models/non_local/README.md)

- etc..

## Requirements

- CUDA 9.2
- cudnn 7.6.5
- Python3.6 (Anaconda)

## Usage

First, clone this repository.

```
git clone http:...
cd maic2020_challenge
```

Next, setup your environment

```
conda env create -f environment.yml
conda activate torch
```

Trainig + Testing with `main.py`. Save resulting file for submission after training.

```
# conv1d_lenet (simplest model)
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfg/conv1d_lenet.yaml

# conv1d_r34_nl4
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfg/conv1d_r34_nl4.yaml

# conv1d_r50_nl5
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfg/conv1d_r50_nl5.yaml

# shufflenet_v2
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfg/shufflenet_v2.yaml

# transformer_basic
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfg/transformer_basic.yaml

# transformer_mtl(MTL: Multi-Task Learning)
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfg/transformer_mtl.yaml
```

## TODO

- [x] pytorch-lightning
- [x] ~~integrate patients info (but not successful)~~
- [x] introduce a reconstruction as pretraining task
  - [ ] two-stage way
    1. train reconstruction model to predict the signal after 5 minutes.
    2. replace decoder of the trained model with classifier from to predict corresponding class.
  - [x] MTL way : linear combination of reconstruction & classification
- [x] focal loss: to remedy class imbalance
- [ ] early stopping
- [ ] lightweight env file

## Supporting Team

Bio-medical Research Institute Center for AI in Medicine(AIM), KNUH(Kyungpook National University Hospital)
