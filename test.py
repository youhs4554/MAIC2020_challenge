from torch.utils.data import TensorDataset, DataLoader
import torch
import os
import numpy as np
import pandas as pd
from models.baseline import BasicConv1d
import tqdm
import torch.nn as nn

if __name__ == "__main__":
    TEST_FILE = "./data/.cache/datasets/MAIC2020/test2_x.csv"
    DATA_ROOT = os.path.join("/data", ".cache", "datasets", "MAIC2020")
    MODEL_PATH = "./experiments/version-14/epoch=1.pth"
    VERSION_DIR = os.path.dirname(MODEL_PATH)

    # test set 로딩
    if os.path.exists(os.path.join(DATA_ROOT, 'x_test.npz')):
        print('loading test...', flush=True, end='')
        test_data = np.load(os.path.join(DATA_ROOT, 'x_test.npz'))['arr_0']
        print('done', flush=True)
    else:
        test_data = pd.read_csv(TEST_FILE).values

        print('saving...', flush=True, end='')
        test_data = np.array(test_data[:, 4:], dtype=np.float32)
        np.savez_compressed(os.path.join(DATA_ROOT, 'x_test.npz'), test_data)
        print('done', flush=True)

    BATCH_SIZE = 1024

    test_data -= 65
    test_data /= 65

    print('filling NANs...', flush=True, end='')
    # nan 을 이전 값으로 채움
    test_data = pd.DataFrame(test_data).fillna(
        method='ffill', axis=1).fillna(method='bfill', axis=1).values
    test_data = torch.from_numpy(test_data).unsqueeze(1)
    print('done', flush=True)

    test_ds = TensorDataset(test_data)
    test_dataloader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=torch.cuda.is_available())
    test_dataloader = iter(test_dataloader)

    # Define model (preproduce baseline implemented at https://github.com/vitaldb/maic2020/blob/master/maic_data_cnn.py)
    # 6-layers 1d-CNNs
    # model = BasicConv1d(dims=[1, 64, 64, 64, 64, 64, 64])

    # from models.resnet1d import resnet18
    # from models.nl_conv1d import NL_Conv1d

    # backbone = resnet18()
    # model = NL_Conv1d(backbone=backbone)

    from models.shufflenet import shufflenet_v2
    model = shufflenet_v2()

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
        x_test, = next(test_dataloader)
        if torch.cuda.is_available():
            x_test = x_test.cuda()

        with torch.no_grad():
            out = model(x_test)
        y_pred += out.flatten().detach().cpu().numpy().tolist()

    np.savetxt(os.path.join(VERSION_DIR, "pred_y.txt"), np.array(y_pred))
