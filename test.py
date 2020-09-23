from torch.utils.data import TensorDataset, DataLoader
import torch
import os
import numpy as np
import pandas as pd
from models.baseline import BasicConv1d
import tqdm
import torch.nn as nn
import joblib

if __name__ == "__main__":
    TEST_FILE = "/data/.cache/datasets/MAIC2020/test2_x.csv"
    DATA_ROOT = os.path.join("/data", ".cache", "datasets", "MAIC2020")
    MODEL_PATH = "./experiments/version-43/epoch=13.pth"
    VERSION_DIR = os.path.dirname(MODEL_PATH)

    # test set 로딩
    if os.path.exists(os.path.join(DATA_ROOT, 'x_test.npz')):
        print('loading test...', flush=True, end='')
        test_data = np.load(os.path.join(DATA_ROOT, 'x_test.npz'))[
            'arr_0'][:, 4:]
        print('done', flush=True)
    else:
        test_data = pd.read_csv(TEST_FILE)

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
        np.savez_compressed(os.path.join(DATA_ROOT, 'x_test.npz'), test_data)
        print('done', flush=True)

    BATCH_SIZE = 512 * torch.cuda.device_count()
    test_data = torch.from_numpy(test_data).float()

    # dataset statistics for input data normalization
    MEAN = 85.15754
    STD = 22.675957

    # apply scaling on raw signals
    test_data = (test_data - MEAN)/STD

    test_ds = TensorDataset(test_data)
    test_dataloader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=torch.cuda.is_available())
    test_dataloader = iter(test_dataloader)

    # Define model (preproduce baseline implemented at https://github.com/vitaldb/maic2020/blob/master/maic_data_cnn.py)
    # 6-layers 1d-CNNs
    # model = BasicConv1d(dims=[1, 64, 64, 64, 64, 64, 64])

    import models.resnet1d
    from models.non_local.nl_conv1d import NL_Conv1d
    backbone = models.resnet1d.resnet18()
    model = NL_Conv1d(backbone=backbone, squad="0,2,2,0", use_ext=False)

    # from models.shufflenet import shufflenet_v2
    # model = shufflenet_v2()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model.cuda()

    print(f"load weights from {MODEL_PATH}...", flush=True, end='')
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print('done', flush=True)

    y_pred = []
    for _ in tqdm.tqdm(range(len(test_dataloader)), desc="Test loop"):
        test_batch = next(test_dataloader)[0]
        x_test = test_batch.unsqueeze(1)

        if torch.cuda.is_available():
            x_test = x_test.cuda()

        with torch.no_grad():
            out = model(x_test, extraction=False)
        y_pred += out.flatten().detach().cpu().numpy().tolist()

    np.savetxt(os.path.join(VERSION_DIR, "pred_y.txt"), np.array(y_pred))
