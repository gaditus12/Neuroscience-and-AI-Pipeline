# Re-import necessary modules after code execution environment reset
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.signal import hilbert, butter, filtfilt

# Redefine signal parameters
fs = 250
t = np.linspace(0, 2, 2 * fs, endpoint=False)

# Simulate realistic EEG-like signal
alpha1 = 0.6 * np.sin(2 * np.pi * 9 * t)
alpha2 = 0.4 * np.sin(2 * np.pi * 10.5 * t)
alpha3 = 0.3 * np.sin(2 * np.pi * 12 * t)
amplitude_modulation = 1 + 0.5 * np.sin(2 * np.pi * 1 * t)
combined_alpha = (alpha1 + alpha2 + alpha3) * amplitude_modulation
theta = 0.2 * np.sin(2 * np.pi * 6 * t)
beta = 0.2 * np.sin(2 * np.pi * 20 * t)
hf_noise = 0.05 * np.random.randn(len(t))
realistic_signal = combined_alpha + theta + beta + hf_noise

# Bandpass filter function
def bandpass_filter(sig, low, high, fs, order=4):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, sig)

# Process signal
filtered_signal = bandpass_filter(realistic_signal, 8, 13, fs)
analytic_signal = hilbert(filtered_signal)
envelope = np.abs(analytic_signal)
median_val = np.median(envelope)
binary_state = (envelope > median_val).astype(int)

# Compute SPI
def compute_spi_from_states(states):
    switches = np.diff(states).nonzero()[0] + 1
    if switches.size < 1:
        return np.nan
    seg_starts = np.r_[0, switches, len(states)]
    durations = np.diff(seg_starts)
    mu, sigma = durations.mean(), durations.std()
    return sigma / mu if mu else np.nan, durations, seg_starts

spi_value, durations, seg_starts = compute_spi_from_states(binary_state)
durations_ms = (durations / fs * 1000).astype(int)

# LaTeX-style SPI equation
spi_text = (
    r"${SPI}_\alpha = \frac{\sigma_{\mathrm{durations}}}{\mu_{\mathrm{durations}}}"
    rf" = \frac{{{np.std(durations):.2f}}}{{{np.mean(durations):.2f}}}"
    rf" = {spi_value:.3f}\quad \in \mathbb{{R}}^{{+}}$"
)

# Plot final figure
plt.figure(figsize=(12, 6))
plt.plot(t, filtered_signal, color='black', label='Filtered Signal (8–13 Hz)', alpha=0.8)
plt.plot(t, envelope, label='Envelope', linewidth=2, color='orange')
plt.axhline(median_val, color='gray', linestyle='--', label='Envelope Median')

plt.fill_between(t, 0, 0.5 * binary_state, color='green', alpha=0.2, label='State = 1 (High Alpha)')
plt.fill_between(t, 0, 0.5 * (1 - binary_state), color='red', alpha=0.2, label='State = 0 (Low Alpha)')

# Annotate run durations
for i, (start_idx, dur_ms) in enumerate(zip(seg_starts[:-1], durations_ms)):
    t_mid = t[start_idx + durations[i] // 2]
    plt.text(t_mid, 0.55, f"{dur_ms} ms", ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=90, alpha=0.8)

# SPI annotation box
plt.text(0.02, 0.87, spi_text, transform=plt.gca().transAxes,
         fontsize=18, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

plt.title('Alpha-Band Envelope and SPI Run Durations')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()
