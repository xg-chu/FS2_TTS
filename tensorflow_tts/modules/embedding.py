import numpy as np
import tensorflow as tf

def get_initializer(initializer_range=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

class Embedding(tf.keras.layers.Embedding):
    """Faster version of embedding."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.int32)
        outputs = tf.gather(self.embeddings, inputs)
        return outputs

class Embeddings(tf.keras.layers.Layer):
    """Construct charactor/phoneme/positional/speaker embeddings."""
    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.encoder_self_attention_params.hidden_size
        self.initializer_range = config.initializer_range
        self.config = config

        self.position_embeddings = Embedding(
            config.max_position_embeddings + 1,
            self.hidden_size,
            weights=[self._sincos_embedding()],
            name="position_embeddings",
            trainable=False,
        )

    def build(self, input_shape):
        """Build shared charactor/phoneme embedding layers."""
        with tf.name_scope("charactor_embeddings"):
            self.charactor_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Get charactor embeddings of inputs.

        Args:
            1. charactor, Tensor (int32) shape [batch_size, length].
            2. speaker_id, Tensor (int32) shape [batch_size]
        Returns:
            Tensor (float32) shape [batch_size, length, embedding_size].

        """
        return self._embedding(inputs, training=training)

    def _embedding(self, inputs, training=False):
        """Applies embedding based on inputs tensor."""
        input_ids, speaker_ids = inputs
        input_shape = tf.shape(input_ids)
        seq_length = input_shape[1]
        position_ids = tf.range(1, seq_length + 1, dtype=tf.int32)[tf.newaxis, :]
        # create embeddings
        inputs_embeds = tf.gather(self.charactor_embeddings, input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # sum embedding
        embeddings = inputs_embeds + tf.cast(position_embeddings, inputs_embeds.dtype)
        return embeddings

    def _sincos_embedding(self):
        position_enc = np.array(
            [
                [
                    pos / np.power(10000, 2.0 * (i // 2) / self.hidden_size)
                    for i in range(self.hidden_size)
                ]
                for pos in range(self.config.max_position_embeddings + 1)
            ]
        )

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        # pad embedding.
        position_enc[0] = 0.0

        return position_enc
