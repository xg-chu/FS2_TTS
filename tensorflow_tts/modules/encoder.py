# FastSpeech2 Encoder
import tensorflow as tf
from tensorflow_tts.modules.FFT import Feed_Forward_Transformer

class Encoder(tf.keras.layers.Layer):
    """Fast Speech encoder module."""
    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.layer = [
            Feed_Forward_Transformer(config, name="layer_._{}".format(i))
            for i in range(config.num_hidden_layers)
        ]

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, attention_mask = inputs
        for layer_module in self.layer:
            hidden_states = layer_module(
                [hidden_states, attention_mask], training=training
            )
        return (hidden_states,)
