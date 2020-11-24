# Duration, Pitch, Energy predictor.
import tensorflow as tf

class Variant_Predictor(tf.keras.layers.Layer):
    """FastSpeech duration predictor module."""
    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_layers = []
        for i in range(config.variant_prediction_num_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(
                    config.variant_predictor_filter,
                    config.variant_predictor_kernel_size,
                    padding="same",
                    name="conv_._{}".format(i),
                )
            )
            self.conv_layers.append(tf.keras.layers.Activation(tf.nn.relu))
            self.conv_layers.append(
                tf.keras.layers.LayerNormalization(
                    epsilon=1e-5, name="LayerNorm_._{}".format(i)
                )
            )
            self.conv_layers.append(
                tf.keras.layers.Dropout(config.variant_predictor_dropout_rate)
            )
        self.conv_layers_sequence = tf.keras.Sequential(self.conv_layers)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        """Call logic."""
        encoder_hidden_states, speaker_ids, attention_mask = inputs
        attention_mask = tf.cast(
            tf.expand_dims(attention_mask, 2), encoder_hidden_states.dtype
        )
        # mask encoder hidden states
        masked_encoder_hidden_states = encoder_hidden_states * attention_mask
        # pass though first layer
        outputs = self.conv_layers_sequence(masked_encoder_hidden_states)
        outputs = self.output_layer(outputs)
        # mask outputs
        masked_outputs = outputs * attention_mask
        masked_outputs = tf.squeeze(masked_outputs, -1)
        return masked_outputs
