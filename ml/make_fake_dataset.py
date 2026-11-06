"""Make a FAKE training table for the Retrofit Monitor ML models.

Your simulation/ folder has ONE wash cycle (32 windows). That is too little
to train on, so this script simulates 100 cycles, 20 of them with a fault.

For every 1-second window it simulates two raw signals, then turns them
into a few numbers (features):

    CT clamp  (current)   -> rms_current_a, peak_current_a, crest_factor
    MPU6050   (vibration) -> vib_rms_g, vib_dom_freq_hz

Run:   python make_fake_dataset.py
Makes: data/fake_features.csv   (one row per 1-second window)

Everything here is invented. The numbers (amps, Hz, g) are plausible
guesses, not measurements, so real accuracy must be re-checked later.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- Settings you can change ----------------
FS = 1000             # samples per second, for both sensors
MAINS_HZ = 50         # Sri Lanka mains frequency
N_NORMAL_CYCLES = 80
N_OVERLOAD_CYCLES = 10
N_UNBALANCED_CYCLES = 10
CURRENT_NOISE_A = 0.03    # random noise on the current signal
VIB_NOISE_G = 0.02        # random noise on the vibration signal
SEED = 42                 # same seed = same file every run

# What each normal state looks like:
#   (current in amps RMS, vibration strength in g RMS, vibration frequency in Hz, seconds)
# Wash and Rinse are close to each other ON PURPOSE: they are hard to tell apart.
STATES = {
    "Idle":  (0.05, 0.01, None, 3),   # None = no real vibration tone, only noise
    "Wash":  (1.80, 0.10, 6.0,  6),
    "Rinse": (1.60, 0.09, 7.0,  5),
    "Drain": (0.60, 0.06, 12.0, 3),
    "Spin":  (2.60, 0.30, 18.0, 6),
}
PROGRAM = ["Idle", "Wash", "Drain", "Rinse", "Drain", "Spin", "Idle"]

rng = np.random.default_rng(SEED)
t = np.arange(FS) / FS     # time axis of ONE window: 0 to 0.999 s


def simulate_window(amps_rms, vib_rms, vib_hz, extra_harmonic=0.0):
    """Make one second of raw current + vibration, then return 5 features."""
    # --- current: a 50 Hz sine wave plus noise ---
    # RMS -> peak: multiply by sqrt(2), the same step as in generate_fake_signal.py
    phase = rng.uniform(0, 2 * np.pi)
    current = amps_rms * np.sqrt(2) * np.sin(2 * np.pi * MAINS_HZ * t + phase)
    current += rng.normal(0, CURRENT_NOISE_A, t.size)

    # --- vibration: a sine wave at the machine's frequency plus noise ---
    # An MPU6050 also reads gravity (about 1 g) as a constant offset.
    vib = 1.0 + rng.normal(0, VIB_NOISE_G, t.size)
    if vib_hz is not None:
        phase = rng.uniform(0, 2 * np.pi)
        vib += vib_rms * np.sqrt(2) * np.sin(2 * np.pi * vib_hz * t + phase)
        if extra_harmonic > 0:     # a second tone at double the frequency
            vib += extra_harmonic * np.sin(2 * np.pi * 2 * vib_hz * t)

    # --- features from the current ---
    rms = np.sqrt(np.mean(current ** 2))
    peak = np.max(np.abs(current))
    crest = peak / rms

    # --- features from the vibration ---
    vib = vib - vib.mean()                          # remove the constant 1 g (gravity)
    vib_rms_g = np.sqrt(np.mean(vib ** 2))
    # FFT: splits the signal into frequencies. Find the strongest one.
    spectrum = np.abs(np.fft.rfft(vib))             # strength at each frequency
    freqs = np.fft.rfftfreq(vib.size, d=1 / FS)     # the frequency of each entry (0, 1, 2 ... Hz)
    dominant = freqs[1:][np.argmax(spectrum[1:])]   # [1:] skips 0 Hz (the constant part)

    return rms, peak, crest, vib_rms_g, dominant


def make_cycle(cycle_id, fault_type):
    """One wash cycle. fault_type is 'none', 'overload' or 'unbalanced'."""
    # Each cycle is slightly different: another day, another load of clothes.
    cur_scale = rng.normal(1.0, 0.06)    # current differs by about 6% from cycle to cycle
    vib_scale = rng.normal(1.0, 0.12)    # vibration differs by about 12%

    # List of (state label, current, vibration, Hz, extra harmonic) per second
    seconds = []
    for name in PROGRAM:
        amps, vib_g, hz, secs = STATES[name]
        secs = max(2, secs + int(rng.integers(-1, 2)))    # each state lasts +/- 1 second
        for _ in range(secs):
            seconds.append([name, amps, vib_g, hz, 0.0])

    # Insert the fault.
    if fault_type == "overload":
        # Motor jammed: huge current after the Spin, strong vibration with a 2x tone.
        spin_end = max(i for i, s in enumerate(seconds) if s[0] == "Spin") + 1
        n = int(rng.integers(3, 6))
        seconds[spin_end:spin_end] = [["Fault", 3.6, 0.20, 15.0, 0.10]] * n
    elif fault_type == "unbalanced":
        # Unbalanced load: the LAST seconds of the Spin shake 3x harder.
        spin_idx = [i for i, s in enumerate(seconds) if s[0] == "Spin"]
        n = int(rng.integers(3, 5))
        for i in spin_idx[-n:]:
            seconds[i] = ["Fault", 2.9, 0.90, 18.0, 0.0]

    rows = []
    for w, (state, amps, vib_g, hz, harm) in enumerate(seconds):
        a = amps * cur_scale * (1 + rng.normal(0, 0.03))      # 3% random change every second
        v = vib_g * vib_scale * (1 + rng.normal(0, 0.05))
        f = None if hz is None else hz + rng.normal(0, 0.4)   # the frequency wanders a little
        rms, peak, crest, vib_rms_g, dom = simulate_window(a, v, f, harm)
        rows.append({
            "cycle_id": cycle_id,
            "window": w,
            "state": state,
            "fault_type": fault_type,
            "rms_current_a": round(rms, 3),
            "peak_current_a": round(peak, 3),
            "crest_factor": round(crest, 3),
            "vib_rms_g": round(vib_rms_g, 4),
            "vib_dom_freq_hz": dom,
        })
    return rows


all_rows = []
cycle_types = (["none"] * N_NORMAL_CYCLES
               + ["overload"] * N_OVERLOAD_CYCLES
               + ["unbalanced"] * N_UNBALANCED_CYCLES)
for cycle_id, ftype in enumerate(cycle_types):
    all_rows += make_cycle(cycle_id, ftype)

df = pd.DataFrame(all_rows)
out = Path(__file__).parent / "data" / "fake_features.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)

print(f"Wrote {len(df)} windows from {df['cycle_id'].nunique()} cycles to {out}")
print()
print(df["state"].value_counts())
