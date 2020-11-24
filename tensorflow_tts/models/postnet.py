import tensorflow as tf

class TacotronPostnet(tf.keras.layers.Layer):
    """Tacotron-2 postnet."""
    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_batch_norm = []
        for i in range(config.n_conv_postnet):
            conv = tf.keras.layers.Conv1D(
                filters=config.postnet_conv_filters
                if i < config.n_conv_postnet - 1
                else config.num_mels,
                kernel_size=config.postnet_conv_kernel_sizes,
                padding="same",
                name="conv_._{}".format(i),
            )
            batch_norm = tf.keras.layers.BatchNormalization(
                axis=-1, name="batch_norm_._{}".format(i)
            )
            self.conv_batch_norm.append((conv, batch_norm))
        self.dropout = tf.keras.layers.Dropout(
            rate=config.postnet_dropout_rate, name="dropout"
        )
        self.activation = [tf.nn.tanh] * (config.n_conv_postnet - 1) + [tf.identity]

    def call(self, inputs, training=False):
        """Call logic."""
        outputs, mask = inputs
        extended_mask = tf.cast(tf.expand_dims(mask, axis=2), outputs.dtype)
        for i, (conv, bn) in enumerate(self.conv_batch_norm):
            outputs = conv(outputs)
            outputs = bn(outputs)
            outputs = self.activation[i](outputs)
            outputs = self.dropout(outputs, training=training)
        return outputs * extended_mask

