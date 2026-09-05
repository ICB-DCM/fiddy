"""Shared test helpers."""

import numpy as np


def deterministic_noise(x: float, amplitude: float = 1.0) -> float:
    """A reproducible stand-in for i.i.d. per-evaluation noise: seeds a
    PRNG from x's own bit pattern, so repeated calls at the same x return
    the same value (unlike real randomness, which would make a test
    non-reproducible), while still behaving like independent noise across
    different x values.

    :param x: The point to derive a deterministic "noise" value from.
    :param amplitude: The noise amplitude; the returned value is uniform
        on ``[-amplitude, amplitude]``.
    :return: A deterministic, reproducible pseudo-noise value.
    """
    bits = np.array([x], dtype=">f8").view(">u8")[0]
    rng = np.random.default_rng(bits)
    return float(rng.uniform(-1, 1)) * amplitude
