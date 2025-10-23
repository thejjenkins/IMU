from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

path = 'imu_data.csv'
df = pd.read_csv(path, header=0).astype(np.float32)
df_original = df.copy()
# header = 0 indicates first row is column names

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

df_acc = pd.DataFrame(df, columns=df.columns[:3])
df_gyr = pd.DataFrame(df, columns=df.columns[3:])
x_axis = df.index.to_numpy()
means_acc = [df_acc[c].mean() for c in df_acc.columns]
stdevs_acc = [df_acc[c].std(ddof=0) for c in df_acc.columns]
means_gyr = [df_gyr[c].mean() for c in df_gyr.columns]
stdevs_gyr = [df_gyr[c].std(ddof=0) for c in df_gyr.columns]
plot_timeseries(df_acc, "Accelerometer (x, y, z)")
plot_timeseries(df_gyr, "Gyroscope (x, y, z)")
plot_normal_distribution(df_acc, means_acc, stdevs_acc, "Accelerometer Normal PDFs (x, y, z)")
plot_normal_distribution(df_gyr, means_gyr, stdevs_gyr, "Gyroscope Normal PDFs (x, y, z)")

scaler = StandardScaler()
X = scaler.fit_transform(df.values)
lowest_bic = np.inf
best_gmm, best_k = None, None
percentile = 0.2 # Anomaly threshold as lower percentile of log-likelihoods
for k in range(1, 10):
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
    gmm.fit(X)
    bic = gmm.bic(X)
    if bic < lowest_bic:
        lowest_bic = bic
        best_gmm = gmm
        best_k = k
loglik = best_gmm.score_samples(X)
scores = pd.Series(loglik, index=df.index, name="log_likelihood")
threshold = np.percentile(scores.values, percentile)
is_anom = scores < threshold
plot_log_likelihood(scores, best_k, percentile)
out = pd.DataFrame(index=df.index)
out["log_likelihood"] = scores
out["is_anomaly"] = is_anom
out_path = 'out.csv'
out.to_csv(out_path, index=True)
print("=== GMM Anomaly Detection Summary ===")
print(f"Input rows: {len(df_original)}")
print(f"Features used: {df.shape[1]}")
print(f"Selected components (BIC): {best_k}")
print(f"Threshold percentile: {percentile}")
print(f"Threshold value: {threshold:.6f}")
print(f"Anomalies flagged: {int(is_anom.sum())}")
print(f"Output CSV: {out_path}")