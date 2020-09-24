#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
from tqdm.contrib.concurrent import process_map  # or thread_map
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import signal
import random
from collections import defaultdict

plt.ioff()


# In[ ]:


# sampling frequency = 100Hz
FS = 100


# In[ ]:


x_train = np.load("/data/.cache/datasets/MAIC2020/x_train.npz")['arr_0'][:, 4:]
y_train = np.load("/data/.cache/datasets/MAIC2020/y_train.npz")["arr_0"]
train_dataset = np.column_stack((x_train, y_train))

x_val = np.load("/data/.cache/datasets/MAIC2020/x_val.npz")['arr_0'][:, 4:]
y_val = np.load("/data/.cache/datasets/MAIC2020/y_val.npz")["arr_0"]
valid_dataset = np.column_stack((x_val, y_val))


# In[ ]:


test_data = np.load(
    "/data/.cache/datasets/MAIC2020/x_test.npz")['arr_0'][:, 4:]
dummy_test_labels = np.empty((len(test_data,)))
dummy_test_labels.fill(-1.0)
test_dataset = np.column_stack((test_data, dummy_test_labels))


# In[ ]:


def _normalize(S, min_level=-20):
    return np.clip((S - min_level) / -min_level, 0, 1)


def save_figure(fig, savePath):
    axes = fig.axes
    for ax in axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    fig.savefig(savePath, bbox_inches='tight', pad_inches=0, dpi=100)


def save_waveImage(sample, save_root_raw, save_root_spec, name, fs=100, nperseg=16, normalize=True):
    # raw signasl image
    fig1 = plt.figure(frameon=False, figsize=(16/9*4, 4))
    ax = fig1.add_subplot(111)
    ax.plot(sample)
    save_figure(fig1, savePath=os.path.join(save_root_raw, f'{name}.png'))

    # spectrogram image
    f, t, Sxx = signal.spectrogram(sample, fs, nperseg=nperseg)

    Sxx = np.clip(Sxx, 1e-6, Sxx.max())
    Sxx = np.log(Sxx)

    if normalize:
        Sxx = _normalize(Sxx)

    fig2 = plt.figure(frameon=False, figsize=(16/9*4, 4))
    ax = fig2.add_subplot(111)
    cf = ax.contourf(t, f, Sxx, 255, cmap='inferno')
    save_figure(fig2, savePath=os.path.join(save_root_spec, f'{name}.png'))

    # close figures
    plt.close(fig1)
    plt.close(fig2)


# In[ ]:


def save_dataset(root, sample, file_id):
    save_root_raw = os.path.join(root, "raw")
    save_root_spec = os.path.join(root, "spec")

    x_sample, y_sample = np.split(sample, [2000])

    label = int(y_sample)

    if label != -1:
        save_root_raw = os.path.join(save_root_raw, str(label))
        save_root_spec = os.path.join(save_root_spec, str(label))

    for p in [save_root_raw, save_root_spec]:
        os.system(f"mkdir -p {p}")

    save_waveImage(x_sample, save_root_raw, save_root_spec,
                   name=file_id, fs=100, nperseg=16, normalize=True)


# In[ ]:


def multi_run_wrapper(args):
    save_dataset(*args)


if __name__ == "__main__":
    # training dataset
    process_map(multi_run_wrapper,
                list(zip(["/data/.cache/datasets/MAIC2020/img_data/train"]*len(train_dataset),
                         train_dataset,
                         range(len(train_dataset)))),
                max_workers=8, chunksize=1, desc="training data...")

    # validation dataset
    process_map(multi_run_wrapper,
                list(zip(["/data/.cache/datasets/MAIC2020/img_data/val"]*len(valid_dataset),
                         valid_dataset,
                         range(len(valid_dataset)))),
                max_workers=8, chunksize=1, desc="validation data...")

    # testing dataset
    process_map(multi_run_wrapper,
                list(zip(["/data/.cache/datasets/MAIC2020/img_data/test"]*len(test_dataset),
                         test_dataset,
                         range(len(test_dataset)))),
                max_workers=8, chunksize=1, desc="testing data...")


# In[ ]:
