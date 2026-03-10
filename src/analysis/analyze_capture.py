# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.20.0",
#     "matplotlib>=3.5.0",
# ]
# ///

import matplotlib
matplotlib.use('TkAgg')

import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data

    except Exception as e:
        print(f"Error: {e}")


def time_domain_plot(sig, t_s, title_suffix=""):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(t_s, sig.real, color='blue', label='I')
    plt.plot(t_s, sig.imag, color='red', label='Q')

    plt.title(f"Time Domain Signal (True Time){title_suffix}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(sig.real, color='blue', label='I')
    plt.plot(sig.imag, color='red', label='Q')

    plt.title(f"Time Domain Signal (Sample Index){title_suffix}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)


def analyze(fname, capture_idx=None):
    # get data
    data = load_data(fname)
    if data is None:
        return

    metadata = data.get("metadata", {})
    mode = metadata.get("mode", "iq_streaming")
    fs_hz = metadata.get("sample_rate", 30000)
    captures = data.get("captures", [])

    print(f"  Mode: {mode}")
    print(f"  Sample rate: {fs_hz} Hz")
    print(f"  Captures: {len(captures)}")

    if mode == "rolling_buffer":
        analyze_rolling_buffer(data, fs_hz, capture_idx)
    else:
        analyze_streaming(data, fs_hz)


def analyze_streaming(data, fs_hz):
    """Analyze old-style streaming captures (128-sample blocks with complex_signal)."""
    [sig, t_s, fs_hz] = parse_streaming_datastruct(data, fs_hz)

    time_domain_plot(sig, t_s)
    spectrogram(sig, fs_hz, window_size=4096, overlap=0)


def analyze_rolling_buffer(data, fs_hz, capture_idx=None):
    """Analyze rolling buffer captures (4096-sample I/Q per trigger event)."""
    captures = data.get("captures", [])

    if capture_idx is not None:
        if capture_idx < 1 or capture_idx > len(captures):
            print(f"  Capture {capture_idx} out of range (1-{len(captures)})")
            return
        captures_to_plot = [(capture_idx, captures[capture_idx - 1])]
    else:
        captures_to_plot = list(enumerate(captures, 1))

    for idx, capture in captures_to_plot:
        sig = get_complex_signal(capture, fs_hz)
        n_samples = len(sig)
        t_s = np.arange(n_samples) / fs_hz

        # Print capture info
        trigger_offset = capture.get("trigger_offset_ms", 0)
        ball_speed = capture.get("ball_speed_mph")
        club_speed = capture.get("club_speed_mph")
        spin_rpm = capture.get("spin_rpm")

        info_parts = [f"trigger@{trigger_offset:.1f}ms"]
        if ball_speed:
            info_parts.append(f"ball={ball_speed}mph")
        if club_speed:
            info_parts.append(f"club={club_speed}mph")
        if spin_rpm:
            info_parts.append(f"spin={spin_rpm}rpm")

        info_str = ", ".join(info_parts)
        print(f"  Capture #{idx}: {n_samples} samples, {n_samples/fs_hz*1000:.1f}ms ({info_str})")

        suffix = f" - Capture #{idx}"
        time_domain_plot(sig, t_s, title_suffix=suffix)
        # Use 128-sample window with 75% overlap for rolling buffer spectrograms
        # to match the processor's overlapping analysis
        spectrogram(sig, fs_hz, window_size=128, overlap=96, title_suffix=suffix)


def get_complex_signal(capture, fs_hz):
    """Build complex I/Q signal from a capture dict, handling both formats."""
    if "complex_signal" in capture:
        return np.array(capture["complex_signal"])

    # Rolling buffer format: raw I/Q samples
    i_raw = np.array(capture["i_samples"], dtype=np.float64)
    q_raw = np.array(capture["q_samples"], dtype=np.float64)

    # Center and scale (12-bit ADC, 3.3V reference)
    i_centered = i_raw - np.mean(i_raw)
    q_centered = q_raw - np.mean(q_raw)
    i_scaled = i_centered * (3.3 / 4096)
    q_scaled = q_centered * (3.3 / 4096)

    return i_scaled + 1j * q_scaled


def parse_streaming_datastruct(data, fs_hz):
    """Parse old streaming format (many 128-sample blocks concatenated)."""
    ts_s = 1/fs_hz
    tmp_sig = []
    tmp_time = []

    captures = data.get("captures")

    for capture in captures:
        sig = get_complex_signal(capture, fs_hz)
        tmp_sig.append(sig)
        t_start = capture.get("sample_time", 0)
        t_end = t_start + (ts_s * (len(sig) - 1))
        tmp_time.append(np.linspace(t_start, t_end, len(sig)))

    sig_tot = np.concatenate(tmp_sig)
    t_tot = np.concatenate(tmp_time)
    t_tot = t_tot - t_tot[0]

    return sig_tot, t_tot, fs_hz


def spectrogram(sig, fs_hz, window_size, overlap, title_suffix=""):
    # full fft for comparison
    sig_fft = np.fft.fftshift(np.fft.fft(sig))
    fft_freqs = np.linspace(-fs_hz/2, fs_hz/2, len(sig), endpoint=False)
    fft_mph = dopp_to_mph(fft_freqs)

    # rough spectral estimate
    hop_size = window_size - overlap
    num_segments = (len(sig) - overlap) // hop_size

    if num_segments < 1:
        print(f"  Warning: signal too short for spectrogram (need > {window_size} samples)")
        return

    window = np.hanning(window_size)

    spec = np.zeros((window_size, num_segments), dtype=complex)

    for i in range(num_segments):
        start = i * hop_size
        segment = sig[start : start + window_size]
        windowed_segment = segment * window
        spectrum = np.fft.fft(windowed_segment)
        spec[:, i] = np.fft.fftshift(spectrum)

    spec_db = db20(spec)
    freqs_hz = np.linspace(-fs_hz/2, fs_hz/2, window_size, endpoint=False)
    freqs_mph = dopp_to_mph(freqs_hz)
    times_ms = np.arange(num_segments) * (hop_size / fs_hz) * 1000

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(spec_db, aspect='auto', origin='lower',
               extent=[times_ms[0], times_ms[-1], freqs_mph[0], freqs_mph[-1]])
    plt.title(f"Spectrogram{title_suffix}")
    plt.ylabel("Speed (mph)")
    plt.xlabel("Time (ms)")
    plt.colorbar(label="dB")

    plt.subplot(1, 2, 2)
    plt.plot(fft_mph, db20_1d(sig_fft))
    plt.title(f"Full FFT{title_suffix}")
    plt.xlabel("Speed (mph)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.tight_layout()


def dopp_to_mph(freqs, fc=24.125e9):
    c = 299792458  # speed of light
    m_per_s = freqs * c / (2 * fc)
    mph = m_per_s * 2.23693629
    return mph


def db20(data):
    return 20 * np.log10(np.abs(data) + 1e-10)


def db20_1d(data):
    return 20 * np.log10(np.abs(data) + 1e-10)


def boldify():
    plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.linewidth': 2.0,
    'lines.linewidth': 2.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'font.size': 16
    })


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Analyze I/Q capture files")
    parser.add_argument("files", nargs="*", help="I/Q capture file(s) to analyze (.pkl)")
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    parser.add_argument("--capture", "-c", type=int, default=None,
                       help="Analyze only capture N (1-indexed, rolling buffer mode only)")
    args = parser.parse_args()

    boldify()

    if args.files:
        for fname in args.files:
            if os.path.exists(fname):
                print(f"Analyzing: {fname}")
                analyze(fname, capture_idx=args.capture)
            elif os.path.exists(os.path.join(args.data_dir, fname)):
                fpath = os.path.join(args.data_dir, fname)
                print(f"Analyzing: {fpath}")
                analyze(fpath, capture_idx=args.capture)
            else:
                print(f"File not found: {fname}")
    else:
        # Default: analyze standard datasets
        data_dir = args.data_dir
        dataset_h1 = "iq_captures_club_swinging.pkl"
        dataset_h0 = "iq_captures_noise.pkl"

        for dataset in [dataset_h0, dataset_h1]:
            fpath = os.path.join(data_dir, dataset)
            if os.path.exists(fpath):
                print(f"Analyzing: {fpath}")
                analyze(fpath, capture_idx=args.capture)
            else:
                print(f"Skipping (not found): {fpath}")

    plt.show()
