import tensorflow as tf

class Length_Regulator(tf.keras.layers.Layer):
    """FastSpeech lengthregulator module."""
    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.config = config

    def call(self, inputs, training=False):
        """Call logic.
        Args:
            1. encoder_hidden_states, Tensor (float32) shape [batch_size, length, hidden_size]
            2. durations_gt, Tensor (float32/int32) shape [batch_size, length]
        """
        encoder_hidden_states, durations_gt = inputs
        outputs, encoder_masks = self._length_regulator(
            encoder_hidden_states, durations_gt
        )
        return outputs, encoder_masks

    def _length_regulator(self, encoder_hidden_states, durations_gt):
        """Length regulator logic."""
        sum_durations = tf.reduce_sum(durations_gt, axis=-1)  # [batch_size]
        max_durations = tf.reduce_max(sum_durations)

        input_shape = tf.shape(encoder_hidden_states)
        batch_size = input_shape[0]
        hidden_size = input_shape[-1]
        # initialize output hidden states and encoder masking.
        outputs = tf.zeros(
            shape=[0, max_durations, hidden_size], dtype=encoder_hidden_states.dtype
        )
        encoder_masks = tf.zeros(shape=[0, max_durations], dtype=tf.int32)

        def condition(
            i,
            batch_size,
            outputs,
            encoder_masks,
            encoder_hidden_states,
            durations_gt,
            max_durations,
        ):
            return tf.less(i, batch_size)

        def body(
            i,
            batch_size,
            outputs,
            encoder_masks,
            encoder_hidden_states,
            durations_gt,
            max_durations,
        ):
            repeats = durations_gt[i]
            real_length = tf.reduce_sum(repeats)
            pad_size = max_durations - real_length
            masks = tf.sequence_mask([real_length], max_durations, dtype=tf.int32)
            repeat_encoder_hidden_states = tf.repeat(
                encoder_hidden_states[i], repeats=repeats, axis=0
            )
            repeat_encoder_hidden_states = tf.expand_dims(
                tf.pad(repeat_encoder_hidden_states, [[0, pad_size], [0, 0]]), 0
            )  # [1, max_durations, hidden_size]
            outputs = tf.concat([outputs, repeat_encoder_hidden_states], axis=0)
            encoder_masks = tf.concat([encoder_masks, masks], axis=0)
            return [
                i + 1,
                batch_size,
                outputs,
                encoder_masks,
                encoder_hidden_states,
                durations_gt,
                max_durations,
            ]

        # initialize iteration i.
        i = tf.constant(0, dtype=tf.int32)
        _, _, outputs, encoder_masks, _, _, _, = tf.while_loop(
            condition,
            body,
            [
                i,
                batch_size,
                outputs,
                encoder_masks,
                encoder_hidden_states,
                durations_gt,
                max_durations,
            ],
            shape_invariants=[
                i.get_shape(),
                batch_size.get_shape(),
                tf.TensorShape(
                    [
                        None,
                        None,
                        self.config.encoder_self_attention_params.hidden_size,
                    ]
                ),
                tf.TensorShape([None, None]),
                encoder_hidden_states.get_shape(),
                durations_gt.get_shape(),
                max_durations.get_shape(),
            ],
        )

        return outputs, encoder_masks