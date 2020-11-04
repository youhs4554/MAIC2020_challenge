import torch
import torch.utils.data
import pandas as pd
import os
import numpy as np
from utils.preprocessing import prepare_data, random_undersampling, raw_signals_to_image_arr
from sklearn.preprocessing import MinMaxScaler
import joblib
import cv2
from PIL import Image
import torchvision.transforms as TF


class MAIC2020(torch.utils.data.Dataset):
    save_dir = os.path.join("/data", ".cache", "datasets", "MAIC2020")
    os.system("mkdir -p {}".format(save_dir))

    def __init__(self,
                 SRATE=100, MINUTES_AHEAD=5, VALIDATION_SPLIT=0.2,
                 transform=None, ext_scaler=None, use_ext=False, train=True, composition=True):

        self.train = train

        # provide composition of target for reconstruction and classification task
        self.composition = composition

        phase = "train" if train else "val"
        if not os.path.exists(os.path.join(self.save_dir, f"x_train.pkl")) or not os.path.exists(os.path.join(self.save_dir, f"x_val.pkl")):
            # split train/validation data
            df = pd.read_csv(os.path.join(self.save_dir, "train_cases.csv"))

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
            xfile_path = os.path.join(self.save_dir, f'x_{phase}.pkl')
            yfile_path = os.path.join(self.save_dir, f'y_{phase}.pkl')

            print('loading...', flush=True, end='')
            self.X = pd.read_pickle(xfile_path).values
            self.y_true = pd.read_pickle(yfile_path).values
            print('done', flush=True)

        # 0: signal_ids, 1~4: externals, 5~: raw_signals
        signal_ids, self.ext, self.X = np.split(self.X, [1, 5, ], axis=1)

        self.y_true = self.y_true[:, [1]].astype(float)

        self.X = self.X.astype("float")
        self.ext = self.ext.astype("float")
        self.signal_ids = signal_ids.ravel()

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

        # expand channel dim
        x_sample = x_sample[:, None]

        if self.transform is not None:
            x_sample = self.transform(x_sample, is_training=self.train)

        # input : 20 [sec] => 20 * (100 [Hz]) = 2,000 [sample points]
        x_sample = torch.tensor(x_sample).view(-1, self.SRATE * 20).float()
        y_sample = torch.tensor(y_sample).long()

        if self.use_ext:
            return x_sample, ext_sample, y_sample

        return x_sample, y_sample


class MAIC2020_rec(MAIC2020):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.future_root = os.path.join(self.save_dir, "future_data")

    def __getitem__(self, ix):
        signal_id = self.signal_ids[ix]

        # load future signal data
        target_X = np.load(os.path.join(self.future_root, signal_id) + ".npy")

        # current signal
        current_X = self.X[ix]

        # expand channel dim
        current_X = current_X[:, None]
        target_X = target_X[:, None]

        if self.transform is not None:
            current_X = self.transform(current_X, is_training=self.train)
            target_X = self.transform(target_X, is_training=self.train)

        # input : 20 [sec] => 20 * (100 [Hz]) = 2,000 [sample points]
        current_X = torch.tensor(current_X).view(-1, self.SRATE * 20).float()

        # target : 60 [sec] after 5min
        target_X = torch.tensor(target_X).view(-1, self.SRATE * 60).float()

        if self.composition:
            # provide classification target along with rec. targets
            y_sample = self.y_true[ix]
            y_sample = torch.tensor(y_sample).float()
            return current_X, target_X, y_sample
        else:
            return current_X, target_X


class MAIC2020_image(MAIC2020):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, ix):
        # current signal
        x_sample = raw_signals_to_image_arr(inputs=self.X[ix])
        x_sample = torch.tensor(x_sample).byte()  # uint8

        # ndarr -> PIL
        x_sample = TF.ToPILImage()(x_sample.permute(2, 0, 1))

        y_sample = self.y_true[ix]
        y_sample = torch.tensor(y_sample).float()

        if self.transform is not None:
            x_sample = self.transform(x_sample)
        else:
            x_sample = TF.ToTensor(x_sample)

        return x_sample, y_sample


class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensors, transform=None):
        assert len(tensors) > 0, "number of tensors is zero"
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        # input to image transformation should be pil image
        x = raw_signals_to_image_arr(inputs=self.tensors[index])
        x = TF.ToPILImage()(x)

        if self.transform is not None:
            x = self.transform(x)
        else:
            x = TF.ToTensor()(x)

        return x

    def __len__(self):
        return len(self.tensors)


def prepare_test_dataset(data_root, transform=None, use_ext=False, use_image=False):
    # test set 로딩
    if os.path.exists(os.path.join(data_root, 'x_test.npz')):
        print('loading test...', flush=True, end='')
        test_data = np.load(os.path.join(data_root, 'x_test.npz'))[
            'arr_0']
        print('done', flush=True)
    else:
        test_data = pd.read_csv(os.path.join(data_root, "test2_x.csv"))

        # one-hot encoding for categorical column
        test_data.sex = test_data.sex.map({"M": 1.0, "F": 0.0})
        test_data = test_data.values

        # first 4 columns are externals
        ext, segx = np.split(test_data, [4, ], axis=1)
        ext_scaler = joblib.load("scaler.gz")

        # crazy way! exclude a categorical column(i.e. sex)
        ext_to_scale = ext[:, [0, 2, 3]]
        # sclaing external data
        ext_to_scale = ext_scaler.transform(ext_to_scale)

        # synthesizes columns: [age, sex, weight, height]
        ext = np.insert(ext_to_scale, 1, ext[:, 1], axis=1)
        # replace first 4 columns with scaled ones
        test_data = np.column_stack((ext, segx))

        print('filling NANs...', flush=True, end='')
        # nan 을 이전 값으로 채움
        test_data = pd.DataFrame(test_data).fillna(
            method='ffill', axis=1).fillna(method='bfill', axis=1).values
        print('done', flush=True)

        print('saving...', flush=True, end='')
        np.savez_compressed(os.path.join(
            data_root, 'x_test.npz'), test_data)
        print('done', flush=True)

    if use_image:
        test_data = torch.from_numpy(test_data).float()
        return CustomTensorDataset(tensors=test_data[:, 4:], transform=transform)
    else:
        if not use_ext:
            test_data = test_data[:, 4:]
            if transform is not None:
                test_data = transform(test_data, is_training=False)
        else:
            if transform is not None:
                test_data[:, 4:] = transform(
                    test_data[:, 4:], is_training=False)

        test_data = torch.from_numpy(test_data).float()
        return torch.utils.data.TensorDataset(test_data)
