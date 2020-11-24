# FastSpeech2 Decoder
import numpy as np
import tensorflow as tf
from tensorflow_tts.modules.FFT import Feed_Forward_Transformer

class Embedding(tf.keras.layers.Embedding):
    """Faster version of embedding."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.int32)
        outputs = tf.gather(self.embeddings, inputs)
        return outputs

class Decoder(tf.keras.layers.Layer):
    """Fast Speech decoder module."""
    def __init__(self, config, is_compatible_encoder, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.is_compatible_encoder = is_compatible_encoder
        # create decoder positional embedding
        self.decoder_positional_embeddings = Embedding(
            config.max_position_embeddings + 1,
            config.hidden_size,
            weights=[self._sincos_embedding()],
            name="position_embeddings",
            trainable=False,
        )
        # layers
        self.layer = [
            Feed_Forward_Transformer(config, name="layer_._{}".format(i))
            for i in range(config.num_hidden_layers)
        ]
        if self.is_compatible_encoder is False:
            self.project_compatible_decoder = tf.keras.layers.Dense(
                units=config.hidden_size, name="project_compatible_decoder"
            )

    def call(self, inputs, training=False):
        hidden_states, speaker_ids, encoder_mask, decoder_pos = inputs
        # project input
        if self.is_compatible_encoder is False:
            hidden_states = self.project_compatible_decoder(hidden_states)
        # calculate new hidden states.
        hidden_states += tf.cast(
            self.decoder_positional_embeddings(decoder_pos), hidden_states.dtype
        )
        for layer_module in self.layer:
            hidden_states = layer_module(
                [hidden_states, encoder_mask], training=training
            )
        return (hidden_states,)

    def _sincos_embedding(self):
        position_enc = np.array(
            [
                [
                    pos / np.power(10000, 2.0 * (i // 2) / self.config.hidden_size)
                    for i in range(self.config.hidden_size)
                ]
                for pos in range(self.config.max_position_embeddings + 1)
            ]
        )

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        # pad embedding.
        position_enc[0] = 0.0
        return position_enc
