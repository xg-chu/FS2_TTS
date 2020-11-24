# Feed-Forward Transformer
import tensorflow as tf
from tensorflow_tts.modules.mh_attention import Attention_Module

def get_initializer(initializer_range=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

class Feed_Forward_Transformer(tf.keras.layers.Layer):
    """Fastspeech module (FFT module on the paper)."""
    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.attention = Attention_Module(config, name="attention")
        self.two_conv1d = Two_Layers_Conv1d(config, name="intermediate")
        self.add_and_norm = Add_and_Norm(config, name="output")

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, attention_mask = inputs

        attention_output = self.attention(
            [hidden_states, attention_mask], training=training
        )[0]

        two_conv1d_output = self.two_conv1d(
            [attention_output, attention_mask], training=training
        )

        fft_output = self.add_and_norm(
            [two_conv1d_output, attention_output], training=training
        )

        masked_layer_output = fft_output * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=fft_output.dtype
        )
        return masked_layer_output

class Two_Layers_Conv1d(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv1d_1 = tf.keras.layers.Conv1D(
            config.intermediate_size,
            kernel_size=config.intermediate_kernel_size,
            kernel_initializer=get_initializer(config.initializer_range),
            padding="same",
            name="conv1d_1",
        )
        self.conv1d_2 = tf.keras.layers.Conv1D(
            config.hidden_size,
            kernel_size=config.intermediate_kernel_size,
            kernel_initializer=get_initializer(config.initializer_range),
            padding="same",
            name="conv1d_2",
        )
        self.intermediate_act_fn = tf.keras.layers.Activation(mish)

    def call(self, inputs):
        """Call logic."""
        hidden_states, attention_mask = inputs

        hidden_states = self.conv1d_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.conv1d_2(hidden_states)

        masked_hidden_states = hidden_states * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=hidden_states.dtype
        )
        return masked_hidden_states

class Add_and_Norm(tf.keras.layers.Layer):
    """Add & Norm module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, name="LayerNorm"
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, input_tensor = inputs

        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

