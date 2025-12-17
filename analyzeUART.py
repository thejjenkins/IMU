from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

path = 'data/2025-12-13_imu_data.csv'
df = pd.read_csv(path, header=0).astype(np.float32) # header = 0 indicates first row is column names
df_original = df.copy()

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

def plot_log_likelihood(scores, k, percentile, threshold, imu):
    plt.hist(scores.values, bins=50)
    plt.axvline(threshold, linestyle="--")
    plt.title(f"Log-likelihood distribution {imu} (GMM, K={k})\nAnomaly threshold at {percentile}th percentile")
    plt.xlabel("log p(x)")
    plt.ylabel("count")
    #plt.tight_layout()
    plt.show()

df_acc = pd.DataFrame(df, columns=df.columns[:3])
df_gyr = pd.DataFrame(df, columns=df.columns[3:])

scaler_acc = StandardScaler().fit(df_acc.values)
X_acc = scaler_acc.transform(df_acc.values)
scaler_gyr = StandardScaler().fit(df_gyr.values)
X_gyr = scaler_gyr.transform(df_gyr.values)
percentile = 1 # Anomaly threshold as lower percentile of log-likelihoods
lowest_bic_acc = np.inf
lowest_bic_gyr = np.inf
best_gmm_acc, best_k_acc = None, None
best_gmm_gyr, best_k_gyr = None, None
biclist_acc = []
biclist_gyr = []
max_k = 10

for k in range(1, max_k): # i want k to be the number of clusters
    gmm = GaussianMixture(n_components=k, covariance_type="spherical", random_state=0)
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
scores_gyr = pd.DataFrame({"gyr_log_likelihood": loglik_gyr}, index=df.index)
scores_acc = pd.DataFrame({"acc_log_likelihood": loglik_acc}, index=df.index)
threshold_acc = np.percentile(scores_acc.values, percentile)
threshold_gyr = np.percentile(scores_gyr.values, percentile)
is_anom_acc = scores_acc < threshold_acc
is_anom_gyr = scores_gyr < threshold_gyr

plot_log_likelihood(scores_acc, best_k_acc, percentile, threshold_acc, imu="Accelerometer")
plot_log_likelihood(scores_gyr, best_k_gyr, percentile, threshold_gyr, imu="Gyroscope")
scores_acc["is_anom_acc"] = is_anom_acc["acc_log_likelihood"]
scores_gyr["is_anom_gyr"] = is_anom_gyr["gyr_log_likelihood"]
scores = pd.concat([scores_acc, scores_gyr], axis=1)
out_path = 'data/out.csv'
scores.to_csv(out_path, index=True)
print("=== GMM Anomaly Detection Summary ===")
print(f"Input rows: {len(df_original)}")
print(f"Features used: {df.shape[1]}")
print(f"Selected components (BIC): Acc = {best_k_acc} clusters, Gyr = {best_k_gyr} clusters")
print(f"Threshold percentile: {percentile}")
print(f"Threshold value accelerometer: {threshold_acc:.6f}")
print(f"Threshold value gyroscope: {threshold_gyr:.6f}")
print(f"Anomalies flagged: {int(is_anom_acc['acc_log_likelihood'].sum()+is_anom_gyr['gyr_log_likelihood'].sum())}")
print(f"Output CSV: {out_path}")