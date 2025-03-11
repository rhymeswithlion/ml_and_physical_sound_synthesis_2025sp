import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy import signal

from data_utils import download_url

DEFAULT_SAMPLE_RATE = 44100


def get_stereo_impulse_response(
    url,
    *,
    length_seconds,
    start_seconds,
    fade_in_seconds,
    fade_out_seconds,
    sr=DEFAULT_SAMPLE_RATE,
):
    from scipy.io import wavfile

    _sr, plate = wavfile.read(download_url(url))

    if _sr != sr:
        pass
        # print(
        #     f"Sampling rate of impulse response ({_sr}) does not match project ({sr})"
        # )

    if plate.dtype == np.int16:
        plate = plate / (2**15)
    elif plate.dtype == np.int32:
        plate = plate / (2**31)
    else:
        raise ValueError(f"Unsupported dtype {plate.dtype}")

    plate_left = plate[:, 0]
    plate_right = plate[:, 1]

    plate_left = plate_left[int(start_seconds * sr) :]
    plate_right = plate_right[int(start_seconds * sr) :]
    plate_left = fade_in(plate_left, fade_in_seconds)
    plate_right = fade_in(plate_right, fade_in_seconds)
    plate_left = plate_left[: int(length_seconds * sr)]
    plate_right = plate_right[: int(length_seconds * sr)]

    plate_left = fade_out(plate_left, fade_out_seconds)
    plate_right = fade_out(plate_right, fade_out_seconds)

    # plt.plot(plate_left)
    # plt.show()
    return plate_left, plate_right


def fade_out(x, seconds, sr=DEFAULT_SAMPLE_RATE):
    # output = x[-int(seconds * sr) :]
    output = x.copy()
    if seconds == 0:
        return output

    fade = np.linspace(1.0, 0.0, int(seconds * sr))
    if len(fade) > len(output):
        fade = fade[-len(output) :]
    output[-len(fade) :] *= fade
    return output


def fade_in(x, seconds, sr=DEFAULT_SAMPLE_RATE):
    output = x.copy()
    if seconds == 0:
        return output
    fade = np.linspace(0.0, 1.0, int(seconds * sr))
    if len(fade) > len(output):
        fade = fade[: len(output)]
    output[: len(fade)] *= fade
    return output


def load_audio(filename):
    import soundfile as sf

    return sf.read(filename)[0]


def play_audio(
    samples,
    sr=DEFAULT_SAMPLE_RATE,
    normalize=False,
    show_analysis=True,
    auto_play=False,
    title=None,
):
    """Play audio samples"""
    import pathlib

    from IPython.display import Audio, display

    # If it's a path, convert to string
    if isinstance(samples, pathlib.Path):
        samples = str(samples)

    # if samples is a file, load it
    if isinstance(samples, str):
        samples = load_audio(samples)

    samples = np.array(samples)

    if len(samples.shape) > 2:
        raise ValueError("samples must be 1D or 2D")

    if title:
        print(title)

    if show_analysis:
        plot_spectrum_and_waveform(samples, sr=sr, title=title)
        # plot_cycles(samples)

    res = Audio(samples, rate=sr, normalize=normalize)

    if auto_play:
        import subprocess
        import tempfile
        import wave

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            with wave.open(f, "wb") as wf:
                num_channels = samples.shape[1] if len(samples.shape) > 1 else 1
                wf.setnchannels(num_channels)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes((samples * 32767).astype(np.int16).tobytes())
            subprocess.run(["afplay", "--volume", "0.5", f.name])
            # time.sleep(0.1)

    display(res)
    # return res


def plot_spectrum_and_waveform(
    samples, sr=DEFAULT_SAMPLE_RATE, max_time=1.0, title=None, oscillations=10
):
    import librosa
    import matplotlib.pyplot as plt

    samples = samples[: int(sr * max_time)]

    S = np.abs(librosa.stft(samples))

    fig = plt.figure(figsize=(15, 4), tight_layout=True)
    if title:
        fig.suptitle(title)
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(
        np.arange(0.0, len(samples[: int(sr * max_time)])) / sr,
        samples[: int(sr * max_time)],
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")

    ax = fig.add_subplot(1, 3, 2)
    librosa.display.specshow(
        librosa.amplitude_to_db(S, ref=np.max),
        x_axis="time",
        ax=ax,
    )

    ax = fig.add_subplot(1, 3, 3)
    zeros = np.where(np.diff(np.sign(samples)) > 0)[0]
    if len(zeros) < oscillations:
        print("Not enough zero crossings")
    else:
        samples = samples[zeros[0] : zeros[oscillations]]
        ax.plot(samples)
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-1, 1)

    plt.show()


def plot_cycles(samples, num_cycles=3):
    import matplotlib.pyplot as plt

    # Find negative to positive crossings
    cycle_starts = np.where(np.diff(np.sign(samples)) > 0)[0]

    # Plot each of these from one cycle start to the next
    fig, axs = plt.subplots(1, num_cycles, figsize=(10, 2), tight_layout=True)

    np.random.seed(0)

    if len(cycle_starts) > 3:
        for ax in axs:
            start_cycle_starts_idx = np.random.choice(range(len(cycle_starts) - 1))
            start = cycle_starts[start_cycle_starts_idx]
            end = cycle_starts[start_cycle_starts_idx + 1]

            overlap = int((end - start) / 20)
            start -= overlap
            end += overlap

            ax.plot(samples[start:end])


@nb.njit
def biquad(
    samples: np.ndarray,
    b0: float,
    b1: float,
    b2: float,
    a0: float,
    a1: float,
    a2: float,
) -> np.ndarray:
    x0, x1, x2 = 0.0, 0.0, 0.0
    y0, y1, y2 = 0.0, 0.0, 0.0

    output = np.zeros(len(samples))

    # normalize all coeffs
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    for i in range(len(samples)):
        x2 = x1
        x1 = x0
        x0 = samples[i]

        y2 = y1
        y1 = y0
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2

        output[i] = y0

    return output


def impulse(secs, sr=DEFAULT_SAMPLE_RATE):
    output = np.zeros(int(sr * secs))
    output[0] = 1
    return output


def delay_samples(samples, delay_length):
    output = np.zeros(len(samples) + delay_length)
    output[delay_length:] = samples

    return output[: len(samples)]


def biquad_lpf_coeffs(f_c, Q, f_s):
    wc = 2 * np.pi * f_c / f_s
    c = 1.0 / np.tan(wc / 2.0)
    phi = c * c
    K = c / Q
    a0 = phi + K + 1.0

    b = [1 / a0, 2.0 / a0, 1.0 / a0]
    a = [1, 2.0 * (1.0 - phi) / a0, (phi - K + 1.0) / a0]

    return np.asarray(b), np.asarray(a)


def plot_freqz(b, a, sr=DEFAULT_SAMPLE_RATE):
    w, h = signal.freqz(b, a, fs=sr)
    fig, ax1 = plt.subplots(tight_layout=True)

    # abs provides the magnitude of the complex values h
    ax1.plot(w, 20 * np.log10(abs(h)), "C0")
    ax1.set_ylabel("Amplitude in dB", color="C0")
    ax1.set(xlabel="Frequency in Hz", xlim=(0, np.pi))
    ax1.set(
        ylim=(
            -24,
            max(0.0, 0.1 + max(20 * np.log10(abs(h)))),
        )
    )

    ax2 = ax1.twinx()

    # angle provides the complex argument
    phase = np.unwrap(np.angle(h))
    ax2.plot(w, phase, "C1")
    ax2.set_ylabel("Phase [rad]", color="C1")
    ax2.grid(True)
    ax2.axis("tight")
    plt.show()


def square_wave(freq, secs, harmonics=51):
    # sampling rate of 44.1 kHz
    sr = 44100

    freq = 440
    secs = 1
    t = np.linspace(0, secs, sr * secs)
    y = np.zeros(sr * secs)

    for harmonic in np.arange(1, harmonics, 2):
        y += 0.15 * np.sin(2 * np.pi * harmonic * freq * t) * 1 / harmonic

    return y


def mfcc_hash(data):
    import json
    from hashlib import md5

    import librosa
    import randomname

    mfcc_values = librosa.feature.mfcc(y=data, sr=DEFAULT_SAMPLE_RATE, n_mfcc=3)
    hashes = [
        md5(
            json.dumps(
                tuple(np.round(mfcc_values[:, i], 0)), separators=(",", ":")
            ).encode()
        ).hexdigest()
        for i in range(mfcc_values.shape[1])
    ]
    return [randomname.get_name(seed=h) for h in hashes]


def white_noise(secs, seed=42, sr=DEFAULT_SAMPLE_RATE):
    np.random.seed(seed)

    # random values between -1 and 1
    return np.random.rand(int(secs * sr)) * 2 - 1


def add_silence(input_signal, silence_secs, sr=DEFAULT_SAMPLE_RATE):
    return np.pad(input_signal, (0, int(silence_secs * sr)))


def comb(input_signal, delay_samples, a, feedback):
    # circular buffer of length delay_samples
    buffer = np.zeros(delay_samples)
    output_signal = np.zeros(len(input_signal))

    filter_state = 0.0

    for i in range(len(input_signal)):
        buffer_index = i % delay_samples
        value = buffer[buffer_index]

        filter_state = value * a + filter_state * (1 - a)
        buffer[buffer_index] = input_signal[i] + filter_state * feedback

        # buffer[buffer_index] = value
        output_signal[i] = value

    return output_signal


@nb.njit
def string(
    input_signal,
    delay_samples,
    feedback,
    b0,
    b1,
    b2,
    a0,
    a1,
    a2,
    apb0,
    apb1,
    apb2,
    apa0,
    apa1,
    apa2,
):
    # circular buffer of length delay_samples
    buffer = np.zeros(delay_samples)
    output_signal = np.zeros(len(input_signal))

    # biquad temp values
    x0, x1, x2 = 0.0, 0.0, 0.0
    y0, y1, y2 = 0.0, 0.0, 0.0

    # all-pass temp values
    ap1x0, ap1x1, ap1x2 = 0.0, 0.0, 0.0
    ap1y0, ap1y1, ap1y2 = 0.0, 0.0, 0.0

    ap2x0, ap2x1, ap2x2 = 0.0, 0.0, 0.0
    ap2y0, ap2y1, ap2y2 = 0.0, 0.0, 0.0

    ap3x0, ap3x1, ap3x2 = 0.0, 0.0, 0.0
    ap3y0, ap3y1, ap3y2 = 0.0, 0.0, 0.0

    ap4x0, ap4x1, ap4x2 = 0.0, 0.0, 0.0
    ap4y0, ap4y1, ap4y2 = 0.0, 0.0, 0.0

    ap5x0, ap5x1, ap5x2 = 0.0, 0.0, 0.0
    ap5y0, ap5y1, ap5y2 = 0.0, 0.0, 0.0

    ap6x0, ap6x1, ap6x2 = 0.0, 0.0, 0.0
    ap6y0, ap6y1, ap6y2 = 0.0, 0.0, 0.0

    # normalize all coeffs
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    apb0 /= apa0
    apb1 /= apa0
    apb2 /= apa0
    apa1 /= apa0
    apa2 /= apa0

    # run an iir
    for i in range(len(input_signal)):
        buffer_index = i % delay_samples

        # biquad
        x2 = x1
        x1 = x0
        x0 = buffer[buffer_index]

        y2 = y1
        y1 = y0
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2

        # all-pass
        ap1x2 = ap1x1
        ap1x1 = ap1x0
        ap1x0 = y0
        ap1y2 = ap1y1
        ap1y1 = ap1y0
        ap1y0 = apb0 * ap1x0 + apb1 * ap1x1 + apb2 * ap1x2 - apa1 * ap1y1 - apa2 * ap1y2
        # all-pass 2
        ap2x2 = ap2x1
        ap2x1 = ap2x0
        ap2x0 = ap1y0
        ap2y2 = ap2y1
        ap2y1 = ap2y0
        ap2y0 = apb0 * ap2x0 + apb1 * ap2x1 + apb2 * ap2x2 - apa1 * ap2y1 - apa2 * ap2y2
        # all-pass 3
        ap3x2 = ap3x1
        ap3x1 = ap3x0
        ap3x0 = ap2y0
        ap3y2 = ap3y1
        ap3y1 = ap3y0
        ap3y0 = apb0 * ap3x0 + apb1 * ap3x1 + apb2 * ap3x2 - apa1 * ap3y1 - apa2 * ap3y2
        # all-pass 4
        ap4x2 = ap4x1
        ap4x1 = ap4x0
        ap4x0 = ap3y0
        ap4y2 = ap4y1
        ap4y1 = ap4y0
        ap4y0 = apb0 * ap4x0 + apb1 * ap4x1 + apb2 * ap4x2 - apa1 * ap4y1 - apa2 * ap4y2
        # all-pass 5
        ap5x2 = ap5x1
        ap5x1 = ap5x0
        ap5x0 = ap4y0
        ap5y2 = ap5y1
        ap5y1 = ap5y0
        ap5y0 = apb0 * ap5x0 + apb1 * ap5x1 + apb2 * ap5x2 - apa1 * ap5y1 - apa2 * ap5y2
        # all-pass 6
        ap6x2 = ap6x1
        ap6x1 = ap6x0
        ap6x0 = ap5y0
        ap6y2 = ap6y1
        ap6y1 = ap6y0
        ap6y0 = apb0 * ap6x0 + apb1 * ap6x1 + apb2 * ap6x2 - apa1 * ap6y1 - apa2 * ap6y2

        # string loop
        value = ap6y0 * feedback + input_signal[i]
        buffer[buffer_index] = value
        output_signal[i] = value

    return output_signal
