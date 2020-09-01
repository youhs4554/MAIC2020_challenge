import numpy as np
import pandas as pd
import os
import tqdm
import torch.utils.data

# 2초 moving average


def moving_average(a, n=200):
    ret = np.nancumsum(a, dtype=np.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def prepare_data(infile='../data/train_cases.csv', save_dir="../data/processed", SRATE=100, MINUTES_AHEAD=5, hot='M'):
    # training set 로딩
    x_train = []  # arterial waveform
    y_train = []  # hypotension

    df_train = pd.read_csv(infile)
    for _, row in tqdm.tqdm(list(df_train.iterrows()), desc="preparing dataset...."):
        caseid = row['caseid']
        age = row['age']
        sex = 1.0 if row['sex'] == hot else 0.0
        weight = row['weight']
        height = row['height']

        if len(x_train) > 15:
            break

        vals = pd.read_csv(os.path.join(os.path.dirname(infile), "train_data", '{}.csv'.format(caseid)),
                           header=None).values.flatten()

        # 20sec (20 00) - 5min (300 00) - 1min (60 00) = 38000 sample
        i = 0
        event_idx = []
        non_event_idx = []
        while i < len(vals) - SRATE * (20 + (1 + MINUTES_AHEAD) * 60):
            segx = vals[i:i + SRATE * 20]
            segy = vals[i + SRATE * (20 + MINUTES_AHEAD * 60)                        :i + SRATE * (20 + (1 + MINUTES_AHEAD) * 60)]

            # 결측값 10% 이상이면
            if np.mean(np.isnan(segx)) > 0.1 or \
                np.mean(np.isnan(segy)) > 0.1 or \
                np.max(segx) > 200 or np.min(segx) < 20 or \
                np.max(segy) > 200 or np.min(segy) < 20 or \
                np.max(segx) - np.min(segx) < 30 or \
                np.max(segy) - np.min(segy) < 30 or \
                (np.abs(np.diff(segx[~np.isnan(segx)])) > 30).any() or \
                    (np.abs(np.diff(segy[~np.isnan(segy)])) > 30).any():
                i += SRATE  # 1 sec 씩 전진
                continue

            # 출력 변수
            segy = moving_average(segy, 2 * SRATE)  # 2 sec moving avg
            event = 1 if np.nanmax(segy) < 65 else 0
            if event:  # event
                event_idx.append(i)
            else:
                non_event_idx.append(i)
            x_train.append(
                [age, sex, weight, height] + segx.tolist())
            y_train.append(event)

            i += 30 * SRATE  # 30sec

        nsamp = len(event_idx) + len(non_event_idx)
        if nsamp > 0:
            print('{}: {} ({:.1f}%)'.format(
                caseid, nsamp, len(event_idx) * 100 / nsamp))

    xfile_path = os.path.join(save_dir,  'x_train.npz')
    yfile_path = os.path.join(save_dir, 'y_train.npz')

    print('saving...\n\n', flush=True, end='')
    print("xfile_path : ", xfile_path)
    print("yfile_path : ", yfile_path)

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=bool)

    np.savez_compressed(xfile_path, x_train)
    np.savez_compressed(yfile_path, y_train)
    print('done', flush=True)

    return x_train, y_train
