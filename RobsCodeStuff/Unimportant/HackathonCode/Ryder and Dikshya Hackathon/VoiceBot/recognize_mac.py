"""
Real-time keyword recognition using an Edge Impulse TFLite model.
macOS version — identical logic to recognize.py with a mic permission check.

Setup (one-time):
    brew install portaudio
    pip install -r requirements.txt

Microphone access: macOS will prompt on first run. If audio silently fails,
go to System Settings > Privacy & Security > Microphone and enable Terminal
(or your IDE / Python launcher).
"""

import collections
import math
import sys
import threading
import time

import librosa
import numpy as np
import sounddevice as sd
from ai_edge_litert.interpreter import Interpreter

# ── CONFIG — edit to match your Edge Impulse project ──────────────────────────
MODEL_PATH = (
    #"ei-hackathon---keyword-detection-transfer-learning-"
    "ei-rob0t-project-1-classifier-tensorflow-lite-float32-model.45.lite"
)

# Labels in model output index order. Edge Impulse alphabetizes by default.
LABELS = ["Go", "Left", "Noise", "Right", "Stop"]

SAMPLE_RATE         = 16000   # Hz — must match EI project setting
AUDIO_LENGTH_MS     = 1000    # ms — must match EI window size
FRAME_LENGTH_MS     = 20      # ms — must match EI MFE setting
FRAME_STRIDE_MS     = 10      # ms — must match EI MFE setting
FFT_SIZE            = 256     # must match EI MFE setting
N_MEL_FILTERS       = 40      # must match EI MFE setting
MEL_FMIN            = 0.0     # Hz — must match EI MFE setting
MEL_FMAX            = 8000.0  # Hz — None in EI means Nyquist = sample_rate / 2

# EI's MFE block uses log(1 + x). If model output is erratic, try 1e-6 instead.
LOG_OFFSET      = 1.0
NOISE_FLOOR_DB  = -52.0   # dB — must match EI MFE setting

INFERENCE_STRIDE_MS  = 500    # ms between inferences (latency vs CPU trade-off)

# Per-class confidence thresholds. Lower values increase sensitivity but also false positives.
THRESHOLDS = {
    "Go":    0.70,
    "Left":  0.50,
    "Noise": 1.00,   # never trigger a detection for Noise
    "Right": 0.45,
    "Stop":  0.70,
}
# ── END CONFIG ────────────────────────────────────────────────────────────────


# Derived constants
AUDIO_SAMPLES     = int(SAMPLE_RATE * AUDIO_LENGTH_MS / 1000)
FRAME_LEN_SAMP    = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)
FRAME_STRIDE_SAMP = int(SAMPLE_RATE * FRAME_STRIDE_MS / 1000)
N_FRAMES          = math.ceil((AUDIO_SAMPLES - FRAME_LEN_SAMP) / FRAME_STRIDE_SAMP) + 1
PADDED_SAMPLES    = (N_FRAMES - 1) * FRAME_STRIDE_SAMP + FRAME_LEN_SAMP
INFERENCE_STRIDE_SAMP = int(SAMPLE_RATE * INFERENCE_STRIDE_MS / 1000)

NOISE_LABEL = "Noise"


def check_microphone():
    """Exit early with a clear message if no input device is available."""
    try:
        info = sd.query_devices(kind='input')
        print(f"Microphone: {info['name']}")
    except sd.PortAudioError as e:
        print(f"ERROR: No microphone found — {e}")
        print("Check System Settings > Privacy & Security > Microphone")
        sys.exit(1)


def load_model(path):
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    inp  = interp.get_input_details()
    outp = interp.get_output_details()

    print("=" * 60)
    print("MODEL INSPECTION")
    for t in inp:
        print(f"  Input : shape={t['shape'].tolist()}  dtype={t['dtype'].__name__}")
    for t in outp:
        print(f"  Output: shape={t['shape'].tolist()}  dtype={t['dtype'].__name__}")

    expected_in  = [1, N_FRAMES * N_MEL_FILTERS]
    expected_out = [1, len(LABELS)]
    actual_in    = inp[0]['shape'].tolist()
    actual_out   = outp[0]['shape'].tolist()

    ok = True
    if actual_in != expected_in:
        print(f"\n  WARNING: input shape {actual_in} != expected {expected_in}")
        print("  Adjust FRAME_LENGTH_MS, FRAME_STRIDE_MS, or N_MEL_FILTERS in CONFIG")
        ok = False
    if actual_out != expected_out:
        print(f"\n  WARNING: output shape {actual_out} != expected {expected_out}")
        print(f"  LABELS has {len(LABELS)} entries but model outputs {actual_out[1]} values")
        ok = False
    if ok:
        print(f"\n  Shapes OK: input {actual_in}, output {actual_out}")
    print("=" * 60)

    return interp, inp[0], outp[0]


def build_filterbank():
    """Mel filterbank matrix (N_MEL_FILTERS, 1 + FFT_SIZE//2). Computed once at startup."""
    fb = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=FFT_SIZE,
        n_mels=N_MEL_FILTERS,
        fmin=MEL_FMIN,
        fmax=MEL_FMAX,
        norm=None,   # EI does NOT use Slaney/area normalization; librosa defaults to it
    )
    return fb.astype(np.float32)


def extract_features(audio_int16, filterbank):
    audio = audio_int16.astype(np.float32) / 32768.0
    audio[1:] -= 0.98 * audio[:-1]

    if len(audio) < PADDED_SAMPLES:
        audio = np.pad(audio, (0, PADDED_SAMPLES - len(audio)))

    # Frame at FRAME_LEN_SAMP stride; when FFT_SIZE < FRAME_LEN_SAMP EI analyzes
    # only the first FFT_SIZE samples of each frame (truncate, not zero-pad).
    idx    = np.arange(FRAME_LEN_SAMP)[None, :] + np.arange(N_FRAMES)[:, None] * FRAME_STRIDE_SAMP
    frames = audio[idx][:, :FFT_SIZE] * np.hanning(FFT_SIZE).astype(np.float32)

    # FFT → power spectrum → noise floor → mel → log
    fft_out    = np.fft.rfft(frames, n=FFT_SIZE)
    power      = fft_out.real ** 2 + fft_out.imag ** 2
    mel_energy = (filterbank @ power.T).T
    mel_energy = np.maximum(mel_energy, 10 ** (NOISE_FLOOR_DB / 10))
    log_mel    = np.log(mel_energy + LOG_OFFSET).astype(np.float32)

    return log_mel.flatten()[np.newaxis, :]


def run_inference(interp, input_detail, output_detail, features):
    interp.set_tensor(input_detail['index'], features)
    interp.invoke()
    return interp.get_tensor(output_detail['index'])[0]


_ring = collections.deque(maxlen=AUDIO_SAMPLES)
_lock = threading.Lock()


def _audio_callback(indata, frames, time_info, status):
    if status:
        print(f"  [audio] {status}")
    samples_i16 = (indata[:, 0] * 32767).astype(np.int16)
    with _lock:
        _ring.extend(samples_i16)


def main():
    check_microphone()

    print(f"Loading model: {MODEL_PATH}")
    interp, input_detail, output_detail = load_model(MODEL_PATH)

    print("Building mel filterbank...")
    filterbank = build_filterbank()

    print(f"Labels: {LABELS}")
    print(f"Thresholds: {THRESHOLDS}  |  Inference every {INFERENCE_STRIDE_MS} ms")
    print("\nListening... (Ctrl+C to stop)\n")

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=INFERENCE_STRIDE_SAMP,
            callback=_audio_callback,
        )
    except sd.PortAudioError as e:
        print(f"ERROR: Could not open microphone — {e}")
        print("Check System Settings > Privacy & Security > Microphone")
        sys.exit(1)

    with stream:
        while True:
            time.sleep(INFERENCE_STRIDE_MS / 1000)

            with _lock:
                if len(_ring) < AUDIO_SAMPLES:
                    continue
                audio_snap = np.array(_ring, dtype=np.int16)

            features = extract_features(audio_snap, filterbank)
            scores   = run_inference(interp, input_detail, output_detail, features)

            best_idx   = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_label = LABELS[best_idx] if best_idx < len(LABELS) else f"class_{best_idx}"

            bar_parts = []
            for i, score in enumerate(scores):
                lbl  = LABELS[i] if i < len(LABELS) else f"cls{i}"
                bar  = "#" * int(score * 20)
                bar_parts.append(f"{lbl}: {bar:<20s} {score:.3f}")
            bar_str = "  ".join(bar_parts)

            detected = [
                LABELS[i] for i, score in enumerate(scores)
                if LABELS[i] != NOISE_LABEL and score >= THRESHOLDS.get(LABELS[i], 0.70)
            ]
            if detected:
                print(f"DETECTED: {', '.join(detected)}  |  {bar_str}")
            else:
                print(f"          {bar_str}", end="\r")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
