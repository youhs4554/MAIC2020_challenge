import numpy as np
import pandas as pd
import os
import tqdm
import torch.utils.data
from imblearn.under_sampling import *
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def raw_signals_to_image_arr(inputs):
    fig = Figure(frameon=False, figsize=(16/9*4, 4))
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.plot(inputs)
    ax.axis('off'), ax.set_xticks([]), ax.set_yticks([])

    fig.tight_layout()
    fig.subplots_adjust(left=0, bottom=0, right=1,
                        top=1, hspace=0, wspace=0)
    canvas.draw()       # draw the canvas, cache the renderer
    foo, (width, height) = canvas.print_to_buffer()

    # Convert to a NumPy array.
    image_arr = np.fromstring(foo, np.uint8).reshape((height, width, 4))
    image_arr = cv2.cvtColor(image_arr, cv2.COLOR_RGBA2RGB)

    return image_arr


def random_undersampling(X_imb, y_imb):

    # inspect class imbalance
    print("[before sampling] Class dist. : ",
          np.unique(y_imb, return_counts=True))

    # random under sampling
    X_samp, y_samp = RandomUnderSampler(
        random_state=0).fit_sample(X_imb, y_imb)

    # inspect class imbalance
    print("[after sampling] Class dist. : ",
          np.unique(y_samp, return_counts=True))

    return X_samp, y_samp


# 2초 moving average


def moving_average(a, n=200):
    ret = np.nancumsum(a, dtype=np.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def prepare_data(df, phase="train", save_dir="../data/processed", SRATE=100, MINUTES_AHEAD=5, hot='M'):
    # to save future data saving
    future_data_dir = os.path.join(save_dir, "future_data")
    os.system(f"mkdir -p {future_data_dir}")

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
            segy = vals[i + SRATE * (20 + MINUTES_AHEAD * 60):i + SRATE * (20 + (1 + MINUTES_AHEAD) * 60)]

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

            # Target sequence (60 sec = 60 * 100 Hz = 6000 samples)
            segy_future = segy

            # 출력 변수
            segy = moving_average(segy, 2 * SRATE)  # 2 sec moving avg
            event = 1 if np.nanmax(segy) < 65 else 0
            signal_id = f"{caseid}_{i//SRATE}"
            future_save_path = os.path.join(future_data_dir, signal_id)

            if event:  # event
                # nan 을 이전 값으로 채움
                segx = pd.DataFrame(segx).fillna(
                    method='ffill', axis=0).fillna(method='bfill', axis=0).values.ravel()
                segy_future = pd.DataFrame(segy_future).fillna(
                    method='ffill', axis=0).fillna(method='bfill', axis=0).values.ravel()

                event_idx.append(i)
                x_data.append(
                    [signal_id, age, sex, weight, height] + segx.tolist())
                # save future signals
                np.save(future_save_path, segy_future)
                y_data.append([signal_id, event])

            elif np.nanmin(segy) > 65:  # non event
                # nan 을 이전 값으로 채움
                segx = pd.DataFrame(segx).fillna(
                    method='ffill', axis=0).fillna(method='bfill', axis=0).values.ravel()
                segy_future = pd.DataFrame(segy_future).fillna(
                    method='ffill', axis=0).fillna(method='bfill', axis=0).values.ravel()

                non_event_idx.append(i)
                x_data.append(
                    [signal_id, age, sex, weight, height] + segx.tolist())
                # save future signals
                np.save(future_save_path, segy_future)
                y_data.append([signal_id, event])

            i += 30 * SRATE  # 30sec

        nsamp = len(event_idx) + len(non_event_idx)
        if nsamp > 0:
            print('{}: {} ({:.1f}%)'.format(
                caseid, nsamp, len(event_idx) * 100 / nsamp))

    xfile_path = os.path.join(save_dir,  f'x_{phase}.pkl')
    yfile_path = os.path.join(save_dir, f'y_{phase}.pkl')

    x_data = pd.DataFrame(x_data, columns=[
                          "ids", "age", "sex", "weight", "height", *[f"mbp_{t}" for t in range(20*SRATE)]])
    y_data = pd.DataFrame(y_data, columns=["ids", "class"])

    print('saving...\n\n', flush=True, end='')
    print("xfile_path : ", xfile_path)
    print("yfile_path : ", yfile_path)
    x_data.to_pickle(xfile_path)
    y_data.to_pickle(yfile_path)
    print('done', flush=True)

    return x_data.values, y_data.values
