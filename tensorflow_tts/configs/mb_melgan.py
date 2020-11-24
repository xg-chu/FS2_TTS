"""Multi-band MelGAN Config object."""

class MBMelGANGenerator_Config():
    """Initialize MelGAN Generator Config."""
    def __init__(self):
        self.kernel_size = 7
        self.out_channels = 4
        self.filters = 384
        self.stacks = 4
        self.upsample_scales = [3, 5, 5]
        self.stack_kernel_size = 3
        self.use_bias = True
        self.nonlinear_activation = "LeakyReLU"
        self.nonlinear_activation_params = {"alpha": 0.2}
        self.padding_type = "REFLECT"
        self.use_final_nolinear_activation = True
        self.is_weight_norm = False
        self.initializer_seed = 42
        self.subbands = 4
        self.taps = 62
        self.cutoff_ratio = 0.142
        self.beta = 9.0
