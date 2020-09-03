import torch
import torch.utils.data
import pandas as pd
# pytorch dataset class implemntations

import os
import numpy as np
from utils.preprocessing import prepare_data


class MAIC2020(torch.utils.data.Dataset):
    save_dir = os.path.join("/data", ".cache", "datasets", "MAIC2020")
    os.system("mkdir -p {}".format(save_dir))

    def __init__(self, infile='data/train_cases.csv', SRATE=100, MINUTES_AHEAD=5, transform=None):
        if len(list(filter(lambda p: os.path.splitext(p)[1] == ".npz", os.listdir(self.save_dir)))) < 2:
            self.X, self.y_true = prepare_data(
                infile, save_dir=self.save_dir,
                SRATE=SRATE, MINUTES_AHEAD=MINUTES_AHEAD)
        else:
            xfile_path = os.path.join(self.save_dir, 'x_train.npz')
            yfile_path = os.path.join(self.save_dir, 'y_train.npz')

            print('loading...', flush=True, end='')
            self.X = np.load(xfile_path)['arr_0']
            self.y_true = np.load(yfile_path)['arr_0']
            print('done', flush=True)

        print('filling NANs...', flush=True, end='')
        # nan 을 이전 값으로 채움
        self.X = pd.DataFrame(self.X).fillna(
            method='ffill', axis=1).fillna(method='bfill', axis=1).values
        print('done', flush=True)

        self.SRATE = SRATE
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        x_sample = self.X[ix]
        y_sample = self.y_true[ix]

        # input : 20 [sec] => 20 * (100 [Hz]) = 2,000 [sample points]
        x_sample = torch.tensor(x_sample).view(-1, self.SRATE * 20).float()
        y_sample = torch.tensor(y_sample).long()

        if self.transform is not None:
            x_sample = self.transform(x_sample)

        return x_sample, y_sample


if __name__ == "__main__":
    pass
