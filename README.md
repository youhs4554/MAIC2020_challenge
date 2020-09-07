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

For trainig the model

```
CUDA_VISIBLE_DEVICES=0 python train.py --...
```

For testing the model

```
CUDA_VISIBLE_DEVICES=0 python test.py --...
```

## TODO

- [ ] pytorch-lightning
- [ ] integrate patients info
- [ ] add unsupervised methods
  - two-steps of training
    1. train reconstruction model to predict the signal after 5 minutes.
    2. replace decoder of the trained model with classifier from to predict corresponding class.
- [ ] simplify env file

## Supporting Team

Bio-medical Research Institute Center for AI in Medicine(AIM), KNUH(Kyungpook National University Hospital)
