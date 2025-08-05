import matplotlib.pyplot as plt
import numpy as np

def plot_time_series_intervals(timestamps, y_true, predictions, lower, upper, title='Time Series Predictions with Intervals'):
    """Visualize time series predictions with conformal intervals"""
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, y_true, label='True Values', color='black', alpha=0.3)
    plt.plot(timestamps, predictions, label='Predictions', color='#1f77b4')
    plt.fill_between(timestamps, lower, upper, color='#1f77b4', alpha=0.3, label=f'Prediction Interval')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()


def plot_coverage_evolution(coverage_rates, window_size=100, title='Coverage Rate Rolling Window'):
    """Plot rolling window coverage rate for adaptive conformal methods"""
    rolling_coverage = np.convolve(coverage_rates, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(12, 4))
    plt.plot(rolling_coverage, label=f'{window_size}-step Rolling Coverage')
    plt.xlabel('Time Step')
    plt.ylabel('Coverage Rate')
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()