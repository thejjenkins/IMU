from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter

path = 'data/2025-11-07_imu_data.csv'
df = pd.read_csv(path, header=0).astype(np.float32) # header = 0 indicates first row is column names
df_original = df.copy()
fs = 100.0  # ODR of IMU in Hz
cutoff_hz = 10.0 # strictly less than Nyquist (fs/2 = 50 Hz)

class RingBuffer:
    def __init__(self, n_streams: int, capacity: int, dtype=np.float32):
        self.buf = np.zeros((n_streams, capacity), dtype=dtype)
        self.capacity = capacity
        self.n_streams = n_streams
        self.head = 0         # next write index
        self.full = False

    def push(self, sample_vec):
        """
        sample_vec: iterable of length n_streams (e.g., [ax, ay, az, gx, gy, gz])
        """
        self.buf[:, self.head] = np.asarray(sample_vec, dtype=self.buf.dtype)
        self.head = (self.head + 1) % self.capacity
        if self.head == 0:
            self.full = True

    def size(self):
        return self.capacity if self.full else self.head

    def view_chronological(self):
        """
        Returns a (n_streams, size) view in chronological order (oldest→newest),
        without copying when possible.
        """
        n = self.size()
        if not self.full:
            return self.buf[:, :n]
        # wrap: concatenate tail and head views
        return np.concatenate((self.buf[:, self.head:], self.buf[:, :self.head]), axis=1)

# Example usage with 6 streams and 128 capacity
rb = RingBuffer(n_streams=6, capacity=128)

# Suppose 'df' is your 6-column DataFrame with numeric data (c1..c6)
for _, row in df.iterrows():
    rb.push(row.values[:6])  # push one 6-element sample

window = rb.view_chronological()  # shape: (6, current_size)
# Now you can run analysis on 'window' (e.g., means per stream):
means = window.mean(axis=1)

def global_range(means, stdevs, k=4, num=1000):
    lowers = [m - k*s for m, s in zip(means, stdevs)]
    uppers = [m + k*s for m, s in zip(means, stdevs)]
    x_min, x_max = float(np.min(lowers)), float(np.max(uppers))
    # If all stds are 0, expand a tiny bit to avoid a degenerate range
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min, x_max = x_min - 1e-6, x_max + 1e-6
    return np.linspace(x_min, x_max, num)

def plot_timeseries(sub_df, title):
    plt.figure()
    for col in sub_df.columns:
        plt.plot(sub_df.index.values, sub_df[col].values, label=col)
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

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
    plt.show()

def plot_log_likelihood(scores, k, percentile):
    plt.hist(scores.values, bins=50)
    plt.axvline(threshold, linestyle="--")
    plt.title(f"Log-likelihood distribution (GMM, K={k})\nAnomaly threshold at {percentile}th percentile")
    plt.xlabel("log p(x)")
    plt.ylabel("count")
    #plt.tight_layout()
    plt.show()

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

df_acc = pd.DataFrame(df, columns=df.columns[:3])
df_gyr = pd.DataFrame(df, columns=df.columns[3:])
acc_body, acc_grav, gyr_filt = preprocess(df_acc.values, df_gyr.values, med_ksize=3)
df_acc_body = pd.DataFrame(acc_body, columns=df_acc.columns)
df_acc_grav = pd.DataFrame(acc_grav, columns=df_acc.columns)
df_gyr_filt = pd.DataFrame(gyr_filt, columns=df_gyr.columns)

scaler_acc = StandardScaler().fit(df_acc_body.values)
X_acc = scaler_acc.transform(df_acc_body.values)
scaler_gyr = StandardScaler().fit(df_gyr_filt.values)
X_gyr = scaler_gyr.transform(df_gyr_filt.values)
percentile = 1 # Anomaly threshold as lower percentile of log-likelihoods
lowest_bic_acc = np.inf
lowest_bic_gyr = np.inf
best_gmm_acc, best_k_acc = None, None
best_gmm_gyr, best_k_gyr = None, None
biclist_acc = []
biclist_gyr = []
max_k = 10

for k in range(1, max_k): # i want k to be the number of clusters
    gmm = GaussianMixture(n_components=k, covariance_type="spherical")
    gmm.fit(X_acc)
    bic = gmm.bic(X_acc)
    biclist_acc.append(bic)
    if bic < lowest_bic_acc:
        lowest_bic_acc = bic
        best_gmm_acc = gmm
        best_k_acc = k
    gmm.fit(X_gyr)
    bic = gmm.bic(X_gyr)
    biclist_gyr.append(bic)
    if bic < lowest_bic_gyr:
        lowest_bic_gyr = bic
        best_gmm_gyr = gmm
        best_k_gyr = k

plt.plot(range(1,max_k), biclist_acc, marker='o')
plt.plot(range(1,max_k), biclist_gyr, marker='o')
plt.legend(['Accelerometer', 'Gyroscope'])
plt.title("BIC vs Number of GMM Components\nSpherical Covariance")
plt.show()

loglik_acc = best_gmm_acc.score_samples(X_acc)
loglik_gyr = best_gmm_gyr.score_samples(X_gyr)
scores = pd.DataFrame({"acc_log_likelihood": loglik_acc, "gyr_log_likelihood": loglik_gyr}, index=df.index)
#scores = pd.Series(loglik, index=df.index, name="log_likelihood")
threshold = np.percentile(scores.values, percentile)
is_anom = scores < threshold

best_k_total = [best_k_acc, best_k_gyr]
for col,k in zip(scores.columns, best_k_total):
    plot_log_likelihood(scores[col], k, percentile)
scores["is_anom_acc"] = is_anom["acc_log_likelihood"]
scores["is_anom_gyr"] = is_anom["gyr_log_likelihood"]
out_path = 'data/out.csv'
scores.to_csv(out_path, index=True)
print("=== GMM Anomaly Detection Summary ===")
print(f"Input rows: {len(df_original)}")
print(f"Features used: {df.shape[1]}")
print(f"Selected components (BIC): Acc = {best_k_acc} clusters, Gyr = {best_k_gyr} clusters")
print(f"Threshold percentile: {percentile}")
print(f"Threshold value: {threshold:.6f}")
print(f"Anomalies flagged: {int(is_anom['acc_log_likelihood'].sum()+is_anom['gyr_log_likelihood'].sum())}")
print(f"Output CSV: {out_path}")