import torch
import torch.utils.data
import pandas as pd
import os
import numpy as np
from utils.preprocessing import prepare_data
from sklearn.preprocessing import MinMaxScaler
import joblib


class MAIC2020(torch.utils.data.Dataset):
    save_dir = os.path.join("/data", ".cache", "datasets", "MAIC2020")
    os.system("mkdir -p {}".format(save_dir))

    def __init__(self,
                 infile='data/train_cases.csv',
                 SRATE=100, MINUTES_AHEAD=5, VALIDATION_SPLIT=0.2,
                 transform=None, ext_scaler=None, use_ext=True, train=True):

        phase = "train" if train else "val"
        if not os.path.exists(os.path.join(self.save_dir, f"x_{phase}.npz")) or not os.path.exists(os.path.join(self.save_dir, f"y_{phase}.npz")):
            # split train/validation data
            df = pd.read_csv(infile)

            # random state is a seed value
            train_df = df.sample(frac=1-VALIDATION_SPLIT, random_state=200)
            valid_df = df.drop(train_df.index)

            x_train, y_train = prepare_data(
                train_df, "train", save_dir=self.save_dir,
                SRATE=SRATE, MINUTES_AHEAD=MINUTES_AHEAD)

            x_val, y_val = prepare_data(
                valid_df, "val", save_dir=self.save_dir,
                SRATE=SRATE, MINUTES_AHEAD=MINUTES_AHEAD)

            self.X = x_train if train else x_val
            self.y_true = y_train if train else y_val

        else:
            xfile_path = os.path.join(self.save_dir, f'x_{phase}.npz')
            yfile_path = os.path.join(self.save_dir, f'y_{phase}.npz')

            print('loading...', flush=True, end='')
            self.X = np.load(xfile_path)['arr_0']
            self.y_true = np.load(yfile_path)['arr_0']
            print('done', flush=True)

        # first 4 columns are externals
        self.ext, self.X = np.split(self.X, [4, ], axis=1)

        if use_ext:
            ext_scaler = MinMaxScaler()
            if ext_scaler is not None:
                ext_scaler = ext_scaler

            # crazy way! exclude a categorical column(i.e. sex)
            ext_to_scale = self.ext[:, [0, 2, 3]]
            # sclaing external data
            ext_to_scale = ext_scaler.fit_transform(ext_to_scale)
            # save scaler as a file
            joblib.dump(ext_scaler, "scaler.gz")

            # synthesizes columns: [age, sex, weight, height]
            self.ext = np.insert(ext_to_scale, 1, self.ext[:, 1], axis=1)

        self.use_ext = use_ext  # use external data or not
        self.SRATE = SRATE
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        if self.use_ext:
            ext_sample = self.ext[ix]
            ext_sample = torch.tensor(ext_sample).float()

        x_sample = self.X[ix]
        y_sample = self.y_true[ix]

        # input : 20 [sec] => 20 * (100 [Hz]) = 2,000 [sample points]
        x_sample = torch.tensor(x_sample).view(-1, self.SRATE * 20).float()
        y_sample = torch.tensor(y_sample).long()

        if self.transform is not None:
            x_sample = self.transform(x_sample)

        if self.use_ext:
            return x_sample, y_sample, ext_sample

        return x_sample, y_sample


if __name__ == "__main__":
    pass
