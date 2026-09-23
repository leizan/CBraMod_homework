"""Shape adapter around the authors' EEGSimpleConv implementation."""
from models.EEGSimpleConv.EEGSimpleConv import EEGSimpleConv


class Model(EEGSimpleConv):
    def __init__(self, params):
        super().__init__(fm=getattr(params, 'simpleconv_fm', 128),
                         n_convs=getattr(params, 'simpleconv_n_convs', 4), resampling=200,
                         kernel_size=getattr(params, 'simpleconv_kernel_size', 8),
                         n_chan=32, n_classes=1, sfreq=200, n_subjects=None)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (32, 4, 200):
            raise ValueError('Expected SHU-MI input [batch, 32, 4, 200].')
        # Concatenates patches in their original temporal order; no rescaling.
        x = x.reshape(x.shape[0], 32, 800)
        # Original CNN returns [B,1]. Preserve the batch dimension when B=1.
        return super().forward(x).squeeze(-1)
