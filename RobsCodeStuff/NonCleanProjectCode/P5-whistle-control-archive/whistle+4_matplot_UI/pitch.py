"""
Core pitch-detection utilities shared by calibrate.py and whistle.py.

Algorithm:
  1. Apply a Hanning window to the audio chunk to reduce spectral leakage.
  2. Compute the real FFT and map bin indices to Hz.
  3. Restrict to the whistle frequency range (config.WHISTLE_MIN/MAX_HZ).
  4. Find the bin with the highest amplitude — that is the dominant pitch.
  5. Return (frequency_hz, rms_amplitude).
     rms < config.AMPLITUDE_THRESHOLD → caller treats it as silence.
"""

import numpy as np
from config import SAMPLE_RATE, CHUNK_SIZE, WHISTLE_MIN_HZ, WHISTLE_MAX_HZ

# Pre-compute the Hanning window and frequency axis once
_WINDOW = np.hanning(CHUNK_SIZE)
_FREQS  = np.fft.rfftfreq(CHUNK_SIZE, 1.0 / SAMPLE_RATE)
_MASK   = (_FREQS >= WHISTLE_MIN_HZ) & (_FREQS <= WHISTLE_MAX_HZ)
_VIS_FREQS = _FREQS[_MASK]


def detect(chunk: np.ndarray):
    """
    chunk : 1-D float32 array of length CHUNK_SIZE, values in [-1, 1].
    Returns (dominant_hz: float, rms: float).
    """
    mono  = chunk[:, 0] if chunk.ndim == 2 else chunk
    rms   = float(np.sqrt(np.mean(mono ** 2)))

    spectrum          = np.abs(np.fft.rfft(mono * _WINDOW))
    whistle_spectrum  = spectrum[_MASK]
    peak_idx          = int(np.argmax(whistle_spectrum))
    dominant_hz       = float(_VIS_FREQS[peak_idx])

    return dominant_hz, rms


def freq_to_command(hz: float, pitch_bands: list) -> str:
    """
    Map a frequency to a command string using the given pitch bands.
    pitch_bands: list of (min_hz, max_hz, command).
    Returns "stop" if no band matches.
    """
    for min_hz, max_hz, command in pitch_bands:
        if min_hz <= hz < max_hz:
            return command
    return "stop"


def spectrum_snapshot(chunk: np.ndarray):
    """
    Returns (frequencies, amplitudes) for the whistle range — used by
    calibrate.py's live display.
    """
    mono     = chunk[:, 0] if chunk.ndim == 2 else chunk
    spectrum = np.abs(np.fft.rfft(mono * _WINDOW))
    return _VIS_FREQS, spectrum[_MASK] / CHUNK_SIZE
