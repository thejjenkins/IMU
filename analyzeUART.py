import csv
import numpy as np
import pandas as pd

df = pd.read_csv('imu_data.csv', header=0).astype(np.float32)  # Assuming the CSV has a header row

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
print(df.info())
