# multi-head self-attention and add&norm module
import tensorflow as tf

def get_initializer(initializer_range=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

class Attention_Module(tf.keras.layers.Layer):
    """Fastspeech attention module."""
    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.self_attention = Self_Attention(config, name="self")
        self.dense_output = SelfOutput(config, name="output")

    def call(self, inputs, training=False):
        input_tensor, attention_mask = inputs

        self_outputs = self.self_attention(
            [input_tensor, attention_mask], training=training
        )
        attention_output = self.dense_output(
            [self_outputs[0], input_tensor], training=training
        )
        masked_attention_output = attention_output * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=attention_output.dtype
        )
        outputs = (masked_attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs

class Self_Attention(tf.keras.layers.Layer):
    """Self attention module."""
    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = self.num_attention_heads * config.attention_head_size

        self.query = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value",
        )

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.config = config

    def transpose_for_scores(self, x, batch_size):
        """Transpose to calculate attention scores."""
        x = tf.reshape(
            x,
            (batch_size, -1, self.num_attention_heads, self.config.attention_head_size),
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, attention_mask = inputs

        batch_size = tf.shape(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(
            tf.shape(key_layer)[-1], attention_scores.dtype
        )  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # extended_attention_masks for self attention encoder.
            extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
            extended_attention_mask = tf.cast(
                extended_attention_mask, attention_scores.dtype
            )
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
            attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.all_head_size))

        return (context_layer,) # (context_layer, attention_probs)

class SelfOutput(tf.keras.layers.Layer):
    """Fastspeech output of self attention module."""
    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, name="LayerNorm"
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, input_tensor = inputs

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
