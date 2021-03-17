"""
This script is copied from https://lexfridman.com/carsync/

@article{fridman2016automated,
  title={Automated synchronization of driving data using vibration and steering events},
  author={Fridman, Lex and Brown, Daniel E and Angell, William and Abdi{'c}, Irman and Reimer, Bryan and Noh, Hae Young},
  journal={Pattern Recognition Letters},
  volume={75},
  pages={9--15},
  year={2016},
  publisher={Elsevier}
}
"""

import numpy as np
from numpy.fft import fft, ifft, fftshift
from statistics import median


def cross_correlation_using_fft(x, y):
    """
    calculate cross correlation using fft

    Args:
        x: signal x
        y: signal y

    Returns:
        float, cross correlation

    """
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)

# shift < 0 means that y starts 'shift' time steps before x # shift > 0 means that y starts 'shift' time steps after x
def compute_shift(c):
    """
    Compute shift given the cross correlation signal

    Args:
        c: np array, cross correlation

    Returns:
        float, shift
            if shift < 0 means y starts 'shift' time steps before x
            if shift = 0 means that y starts 'shift' time steps after x
    """
    zero_index = int(len(c) / 2) - 1
    shift = zero_index - np.argmax(abs(c - median(c)))
    return shift

