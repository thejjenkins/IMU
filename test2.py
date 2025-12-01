import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter

path = 'data/2025-11-07_imu_data.csv'
df = pd.read_csv(path, header=0).astype(np.float32)
fs = 100.0  # ODR of IMU in Hz
cutoff_hz = 10.0 # strictly less than Nyquist (fs/2 = 50 Hz)

def lpf_butter(x, cutoff_hz, order=3):
    b, a = butter(order, cutoff_hz, btype='low', fs=fs)
    return filtfilt(b, a, x, axis=0)

def preprocess(acc_xyz, gyr_xyz, med_ksize=3):
    # 1) median filter (odd ksize; paper didn’t fix the size)
    acc_m = median_filter(acc_xyz, size=(med_ksize, 1), mode="reflect")
    gyr_m = median_filter(gyr_xyz, size=(med_ksize, 1), mode="reflect")

    # 2) 3rd-order LPF @ 10 Hz (noise reduction)
    acc_lp = lpf_butter(acc_m, cutoff_hz, order=3)
    gyr_lp = lpf_butter(gyr_m, cutoff_hz, order=3)

    # 3) gravity/body split (Butterworth LPF @ 0.3 Hz; order not specified)
    grav = lpf_butter(acc_lp, 0.3, order=3)   # order=3 is a common choice
    body = acc_lp - grav
    return body, grav, gyr_lp

def plot_timeseries(sub_df, title):
    plt.figure()
    colors = ['red', 'green', 'blue']
    for col,color in zip(sub_df.columns,colors):
        plt.plot(sub_df.index.values, sub_df[col].values, label=col, color=color)
    # plt.plot(df_acc_body.index.values, df_acc_body['Accel_X'].values, label='Butterworth X', color='yellow')
    # plt.plot(df_acc_body.index.values, df_acc_body['Accel_Y'].values, label='Butterworth Y', color='orange')
    # plt.plot(df_acc_body.index.values, df_acc_body['Accel_Z'].values, label='Butterworth Z', color='brown')
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

def global_range(means, stdevs, k=4, num=1000):
    lowers = [m - k*s for m, s in zip(means, stdevs)]
    uppers = [m + k*s for m, s in zip(means, stdevs)]
    x_min, x_max = float(np.min(lowers)), float(np.max(uppers))
    # If all stds are 0, expand a tiny bit to avoid a degenerate range
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min, x_max = x_min - 1e-6, x_max + 1e-6
    return np.linspace(x_min, x_max, num)

def plot_normal_distribution(sub_df, means, stand_devs, title):
    x_axis = global_range(means, stand_devs, k=4, num=sub_df.shape[0])
    plt.figure()
    for c, m, s in zip(sub_df.columns, means, stand_devs):
        y = norm.pdf(x_axis, loc=m, scale=s)
        plt.plot(x_axis, y, label=f'{c}: μ={m:.3g}, σ={s:.3g}')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    #plt.show()

df_acc = pd.DataFrame(df, columns=df.columns[:3])
df_gyr = pd.DataFrame(df, columns=df.columns[3:])
acc_body, acc_grav, gyr_filt = preprocess(df_acc.values, df_gyr.values, med_ksize=3)
df_acc_body = pd.DataFrame(acc_body, columns=df_acc.columns)
df_acc_grav = pd.DataFrame(acc_grav, columns=df_acc.columns)
df_gyr_filt = pd.DataFrame(gyr_filt, columns=df_gyr.columns)
means_acc = [df_acc[c].mean() for c in df_acc.columns]
stdevs_acc = [df_acc[c].std(ddof=0) for c in df_acc.columns]
means_gyr = [df_gyr[c].mean() for c in df_gyr.columns]
stdevs_gyr = [df_gyr[c].std(ddof=0) for c in df_gyr.columns]
means_acc_body = [df_acc_body[c].mean() for c in df_acc_body.columns]
stdevs_acc_body = [df_acc_body[c].std(ddof=0) for c in df_acc_body.columns]
means_acc_grav = [df_acc_grav[c].mean() for c in df_acc_grav.columns]
stdevs_acc_grav = [df_acc_grav[c].std(ddof=0) for c in df_acc_grav.columns]
means_gyr_filt = [df_gyr_filt[c].mean() for c in df_gyr_filt.columns]
stdevs_gyr_filt = [df_gyr_filt[c].std(ddof=0) for c in df_gyr_filt.columns]
plot_normal_distribution(df_acc, means_acc, stdevs_acc, "Accelerometer Normal PDFs (x, y, z)")
plot_normal_distribution(df_gyr, means_gyr, stdevs_gyr, "Gyroscope Normal PDFs (x, y, z)")
plot_normal_distribution(df_acc_body, means_acc_body, stdevs_acc_body, "Accelerometer Body Normal PDFs (x, y, z)")
plot_normal_distribution(df_acc_grav, means_acc_grav, stdevs_acc_grav, "Accelerometer Grav Normal PDFs (x, y, z)")
plot_normal_distribution(df_gyr_filt, means_gyr_filt, stdevs_gyr_filt, "Gyroscope Filtered Normal PDFs (x, y, z)")
plot_timeseries(df_acc, "Accelerometer Original")
plot_timeseries(df_gyr, "Gyroscope Original")
plot_timeseries(df_acc_body, "Accelerometer Filtered Body Component")
plot_timeseries(df_acc_grav, "Accelerometer Filtered Gravity Component")
plot_timeseries(df_gyr_filt, "Gyroscope Filtered")

plt.show()