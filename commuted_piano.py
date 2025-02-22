import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.signal import convolve

from audio_utils import (
    add_silence,
    biquad_lpf_coeffs,
    comb,
    get_stereo_impulse_response,
    impulse,
    mfcc_hash,
    play_audio,
    string,
    white_noise,
)


def calculate_strike(length_samples, strike_param_set):
    result = np.zeros(length_samples)
    for strike_params in strike_param_set:
        x = np.linspace(strike_params[0], strike_params[1], length_samples)
        strike = np.exp(-10 * (1 - np.pow(7, -0.75 - x)) ** 2)
        strike /= sum(strike)
        result += strike

    return result


def mtof(m):
    return 440 * 2 ** ((m - 69) / 12)


class _ModelParams(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        # arbitrary_types_allowed=True
    )


class LPFParams(_ModelParams):
    freq: float = 17000
    q: float = 0.2


class StrikeParams(_ModelParams):
    left: float = -1
    right: float = 10.0


class StringParams(_ModelParams):
    gain: float
    lpf_params: LPFParams = LPFParams()
    ap_denom: list[float]
    freq_ratio: float
    feedback_gain: float


class ImpulseResponseParams(_ModelParams):
    start_seconds: float
    length_seconds: float
    fade_in_seconds: float
    fade_out_seconds: float
    url: str
    dry_wet_balance: float


class CommutedPianoParams(_ModelParams):
    freq: float
    channel: str = "L"
    # lpf_params: LPFParams = LPFParams()
    # ap_denom: list[float]
    sample_rate: int = 44100
    hammer_delay_ms: float
    strike_params_list: list[StrikeParams]
    strike_num_samples: int
    ir_params: ImpulseResponseParams

    strings: list[StringParams]

    def to_piano_note(self, show_graphs=False):
        return params_to_piano_note(self, show_graphs=show_graphs)


def hammer_delay(input_signal, delay_samples):
    # print(delay_samples)
    if len(input_signal) < delay_samples:
        input_signal = np.pad(input_signal, (0, delay_samples - len(input_signal) + 1))
    result = np.roll(input_signal, delay_samples)
    result[:delay_samples] = 0
    result += -1 * input_signal
    return result


def add_with_padding(a, b):
    # print(len(a), len(b))
    if len(a) < len(b):
        a = np.pad(a, (0, len(b) - len(a)))
    else:
        b = np.pad(b, (0, len(a) - len(b)))

    return a + b


def params_to_piano_note(p: CommutedPianoParams, *, show_graphs=False):
    from functools import reduce

    freq = p.freq
    channel = p.channel

    strike = calculate_strike(
        p.strike_num_samples,
        [
            (strike_params.left, strike_params.right)
            for strike_params in p.strike_params_list
        ],
    )

    plate_left, plate_right = get_stereo_impulse_response(
        start_seconds=p.ir_params.start_seconds,
        length_seconds=p.ir_params.length_seconds,
        fade_in_seconds=p.ir_params.fade_in_seconds,
        fade_out_seconds=p.ir_params.fade_out_seconds,
        url=p.ir_params.url,
    )
    plates = {"L": plate_left, "R": plate_right}
    # print("plate len", len(plate_left))
    # assert len(plate_left) == 25000

    # single impulse
    res = impulse(0.001)
    # print(mfcc_hash(res)[:3])

    # hammer delay
    # res = hammer_delay(res, int(p.sample_rate * p.hammer_delay_ms / 1000))

    # white noise
    res = convolve(res, white_noise(0.05))
    res /= np.max(np.abs(res))

    if show_graphs:
        print("white_noise", mfcc_hash(res)[:3])
        play_audio(0.25 * res, title="white_noise")

    for i in range(10):
        res = comb(res, int((5 + i) / 3), 0.95 - 0.03 * i, 0.10)
    res /= np.max(np.abs(res))

    if show_graphs:
        print("many filters", mfcc_hash(res)[:3])
        play_audio(0.25 * res, title="many filters")

    res = convolve(res, strike)
    res /= np.max(np.abs(res))

    if show_graphs:
        print("strike", mfcc_hash(res)[:3])
        play_audio(0.25 * res, title="strike")

    res = add_with_padding(
        np.cos(np.pi / 2.0 * (1 - p.ir_params.dry_wet_balance))
        * convolve(plates[channel], res),
        np.cos(np.pi / 2.0 * (p.ir_params.dry_wet_balance)) * res,
    )
    res /= np.max(np.abs(res))

    if show_graphs:
        print("plate", mfcc_hash(res)[:3])
        play_audio(0.25 * res, title="plate")

    res = reduce(
        lambda x, y: x + y,
        [
            string_params.gain
            * string(
                add_silence(res, 5),
                int(p.sample_rate / freq / string_params.freq_ratio),
                string_params.feedback_gain,
                *(
                    (
                        _b_and_a := biquad_lpf_coeffs(
                            string_params.lpf_params.freq,
                            string_params.lpf_params.q,
                            p.sample_rate,
                        )
                    )[0]
                ),
                *(_b_and_a[1]),
                *(ap_denom := np.array(string_params.ap_denom))[::-1],
                *ap_denom,
            )
            for string_params in p.strings
        ],
    )
    # print(_b_and_a)

    res /= np.max(np.abs(res))

    if show_graphs:
        print("string", mfcc_hash(res)[:3])
        play_audio(0.25 * res, title="string")

    if show_graphs:
        print("final", mfcc_hash(res)[:3])
        play_audio(0.25 * res, auto_play=True, title="final")

    res /= np.max(np.abs(res))

    first_nonzero = np.argmax(np.abs(res) > 0.01)
    last_nonzero = len(res) - np.argmax(np.abs(res[::-1]) > 0.01)
    res = res[first_nonzero:last_nonzero]

    return res * 0.95  # avoid clipping


def midi_to_piano_note(m, show_graphs=False, channel="L"):
    freq = mtof(m)

    return params_to_piano_note(freq=freq, show_graphs=show_graphs, channel=channel)
