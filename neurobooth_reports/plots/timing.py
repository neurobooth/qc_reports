"""
Plotting functions relating to timing and synchronization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve


def time_offset_plot(
        ax: plt.Axes,
        ts1: np.ndarray,
        ts2: np.ndarray,
        ts1_name: str = 'Time 1',
        ts2_name: str = 'Time 2'
) -> None:
    diff = ts2 - ts1
    ax.plot(diff, linewidth=1)
    ax.axhline(diff.mean(), linestyle='--', linewidth=0.5, color='k')
    ax.set_xlabel('Sample #')
    ax.set_ylabel(f'{ts2_name} - {ts1_name}')


def time_stability_plot(ax: plt.Axes, ts: np.ndarray, window_width_sec: float = 1) -> None:
    ts_diff = np.diff(ts)
    n_sample = int(round(1 / ts_diff.mean()) * window_width_sec)
    if n_sample % 2 == 0:  # Want odd window for convolution
        n_sample += 1

    # Compute centered moving average via convolution with a uniform window that sums to 1
    # Some artifacts will exist for (n_sample / 2) samples at the endpoints due to zero-padding
    window = np.ones(n_sample) / n_sample
    ts_diff_smoothed = convolve(ts_diff, window, mode='same')
    ax.plot(ts_diff_smoothed, linewidth=1)

    # Shade the region corresponding to convolution artifacts
    half_window = n_sample // 2
    N = ts_diff_smoothed.shape[0] - 1
    ax.axvspan(0, half_window, color='r', alpha=0.3)
    ax.axvspan(N - half_window, N, color='r', alpha=0.3)

    ax.set_xlabel('Sample No.')
    ax.set_ylabel('Moving Avg. of $\\Delta t$')
    ax.set_xlim(0, N)
