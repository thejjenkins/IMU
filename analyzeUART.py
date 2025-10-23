from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

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

df_acc = pd.DataFrame(df, columns=df.columns[:3])
df_gyr = pd.DataFrame(df, columns=df.columns[3:])
x_axis = df.index.to_numpy()
means_acc = [df_acc[c].mean() for c in df_acc.columns]
stdevs_acc = [df_acc[c].std(ddof=0) for c in df_acc.columns]
means_gyr = [df_gyr[c].mean() for c in df_gyr.columns]
stdevs_gyr = [df_gyr[c].std(ddof=0) for c in df_gyr.columns]
plt.figure()
plt.plot(x_axis, df_acc.values)
plt.xlabel("Sample index")
plt.ylabel("Value")
plt.legend(df_acc.columns)
#plt.show()
plt.figure()
plt.plot(x_axis, df_gyr.values)
plt.xlabel("Sample index")
plt.ylabel("Value")
plt.legend(df_gyr.columns)
#plt.show()

def global_range(means, stdevs, k=4, num=1000):
    lowers = [m - k*s for m, s in zip(means, stdevs)]
    uppers = [m + k*s for m, s in zip(means, stdevs)]
    x_min, x_max = float(np.min(lowers)), float(np.max(uppers))
    # If all stds are 0, expand a tiny bit to avoid a degenerate range
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min, x_max = x_min - 1e-6, x_max + 1e-6
    return np.linspace(x_min, x_max, num)
x_acc = global_range(means_acc, stdevs_acc, k=4, num=df_acc.shape[0])
plt.figure()
for c, m, s in zip(df_acc.columns, means_acc, stdevs_acc):
    y = norm.pdf(x_acc, loc=m, scale=s)
    plt.plot(x_acc, y, label=f'{c}: μ={m:.3g}, σ={s:.3g}')
plt.title('Accelerometer normal PDFs (x, y, z)')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
#plt.show()
x_gyr = global_range(means_gyr, stdevs_gyr, k=4, num=df_gyr.shape[0])
plt.figure()
for c, m, s in zip(df_gyr.columns, means_gyr, stdevs_gyr):
    y = norm.pdf(x_gyr, loc=m, scale=s)
    plt.plot(x_gyr, y, label=f'{c}: μ={m:.3g}, σ={s:.3g}')

plt.title('Gyroscope normal PDFs (x, y, z)')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()