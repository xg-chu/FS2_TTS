"""FastSpeech2 Config object."""

class FastSpeech2_Config(object):
    """Initialize FastSpeech Config."""

    def __init__(self):
        # general params
        self.vocab_size = 219 # baker dataset
        self.initializer_range = 0.02
        self.max_position_embeddings = 2048
        # encoder params
        self.encoder_self_attention_params = Self_Attention_Configs(
            initializer_range=self.initializer_range, max_position_embeddings=self.max_position_embeddings)
        # variant predictor params
        self.variant_predictor_params = Variant_Predictor_Configs(initializer_range=self.initializer_range)
        # decoder params
        self.decoder_self_attention_params = Self_Attention_Configs(
            initializer_range=self.initializer_range, max_position_embeddings=self.max_position_embeddings)
        # postnet
        self.num_mels = 80
        self.n_conv_postnet = 5
        self.postnet_conv_filters = 512
        self.postnet_conv_kernel_sizes = 5
        self.postnet_dropout_rate = 0.1

class Self_Attention_Configs():
    def __init__(self, initializer_range=0.02, max_position_embeddings=2048):
        self.hidden_size = 256
        self.num_hidden_layers = 3
        self.num_attention_heads = 2
        self.attention_head_size = 16
        self.intermediate_size = 1024
        self.intermediate_kernel_size = 3
        self.initializer_range = initializer_range
        self.hidden_dropout_prob = 0.2
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = max_position_embeddings

class Variant_Predictor_Configs():
    def __init__(self, initializer_range=0.02):
        self.variant_predictor_filter = 256
        self.variant_prediction_num_conv_layers = 2
        self.variant_predictor_kernel_size = 3
        self.variant_predictor_dropout_rate = 0.5
        self.initializer_range = initializer_range
