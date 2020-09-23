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


def prepare_data(df, phase="train", save_dir="../data/processed", SRATE=100, MINUTES_AHEAD=5, hot='M'):
    # training set 로딩
    x_data = []  # arterial waveform
    y_data = []  # hypotension

    for _, row in tqdm.tqdm(list(df.iterrows()), desc=f"preparing {phase}ing dataset...."):
        caseid = row['caseid']
        age = row['age']
        sex = 1.0 if row['sex'] == hot else 0.0
        weight = row['weight']
        height = row['height']

        vals = pd.read_csv(os.path.join(save_dir, "train_data", '{}.csv'.format(caseid)),
                           header=None).values.flatten()

        # 앞 뒤의 결측값을 제거
        case_valid_mask = ~np.isnan(vals)
        vals = vals[(np.cumsum(case_valid_mask) != 0) & (
            np.cumsum(case_valid_mask[::-1])[::-1] != 0)]

        if np.nanmax(vals) < 120:
            print('mbp < 120')
            continue

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
                np.nanmax(segx) > 200 or np.nanmin(segx) < 20 or \
                np.nanmax(segy) > 200 or np.nanmin(segy) < 20 or \
                np.nanmax(segx) - np.nanmin(segx) < 30 or \
                np.nanmax(segy) - np.nanmin(segy) < 30 or \
                (np.abs(np.diff(segx[~np.isnan(segx)])) > 30).any() or \
                    (np.abs(np.diff(segy[~np.isnan(segy)])) > 30).any():
                i += SRATE  # 1 sec 씩 전진
                continue

            # 출력 변수
            segy = moving_average(segy, 2 * SRATE)  # 2 sec moving avg
            event = 1 if np.nanmax(segy) < 65 else 0
            if event:  # event
                event_idx.append(i)
                x_data.append(
                    [age, sex, weight, height] + segx.tolist())
                y_data.append(event)

            elif np.nanmin(segy) > 65:  # non event
                non_event_idx.append(i)
                x_data.append(
                    [age, sex, weight, height] + segx.tolist())
                y_data.append(event)

            i += 30 * SRATE  # 30sec

        nsamp = len(event_idx) + len(non_event_idx)
        if nsamp > 0:
            print('{}: {} ({:.1f}%)'.format(
                caseid, nsamp, len(event_idx) * 100 / nsamp))

    xfile_path = os.path.join(save_dir,  f'x_{phase}.npz')
    yfile_path = os.path.join(save_dir, f'y_{phase}.npz')

    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=bool)

    print('filling NANs...', flush=True, end='')
    # nan 을 이전 값으로 채움
    x_data = pd.DataFrame(x_data).fillna(
        method='ffill', axis=1).fillna(method='bfill', axis=1).values
    print('done', flush=True)

    print('saving...\n\n', flush=True, end='')
    print("xfile_path : ", xfile_path)
    print("yfile_path : ", yfile_path)
    np.savez_compressed(xfile_path, x_data)
    np.savez_compressed(yfile_path, y_data)
    print('done', flush=True)

    return x_data, y_data
