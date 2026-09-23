"""Same-rate resampling compatibility for already-preprocessed SHU-MI.

Original torchaudio.transforms.Resample returns its input when rates match.
Only this case is supported here; no alternative resampling algorithm is used.
"""
from torch import nn


def Resample(orig_freq, new_freq):
    if orig_freq != new_freq:
        raise ValueError('This SHU-MI adapter requires equal input/output sample rates.')
    return nn.Identity()
