"""Tensorflow Model modules for FastSpeech2."""
import numpy as np
import tensorflow as tf

from tensorflow_tts.modules import (
    Encoder, Embeddings, Variant_Predictor,
    Length_Regulator, Decoder)

from tensorflow_tts.models.postnet import TacotronPostnet

class FastSpeech2(tf.keras.Model):
    """TF Fastspeech module."""
    def __init__(self, config, **kwargs):
        """Init layers for fastspeech."""
        super().__init__(**kwargs)
        self.embeddings = Embeddings(config, name="embeddings")
        self.encoder = Encoder(
            config.encoder_self_attention_params, name="encoder"
        )
        self.length_regulator = Length_Regulator(
            config,
            name="length_regulator",
        )
        self.decoder = Decoder(
            config.decoder_self_attention_params,
            is_compatible_encoder=config.encoder_self_attention_params.hidden_size
            == config.decoder_self_attention_params.hidden_size,
            name="decoder",
        )
        self.mel_dense = tf.keras.layers.Dense(
            units=config.num_mels, dtype=tf.float32, name="mel_before"
        )
        self.postnet = TacotronPostnet(
            config=config, dtype=tf.float32, name="postnet"
        )
        self.f0_predictor = Variant_Predictor(
            config.variant_predictor_params, dtype=tf.float32, name="f0_predictor"
        )
        self.energy_predictor = Variant_Predictor(
            config.variant_predictor_params, dtype=tf.float32, name="energy_predictor",
        )
        self.duration_predictor = Variant_Predictor(
            config.variant_predictor_params, dtype=tf.float32, name="duration_predictor"
        )
        # define f0_embeddings and energy_embeddings
        self.f0_embeddings = tf.keras.layers.Conv1D(
            filters=config.encoder_self_attention_params.hidden_size,
            kernel_size=9,
            padding="same",
            name="f0_embeddings",
        )
        self.f0_dropout = tf.keras.layers.Dropout(0.5)
        self.energy_embeddings = tf.keras.layers.Conv1D(
            filters=config.encoder_self_attention_params.hidden_size,
            kernel_size=9,
            padding="same",
            name="energy_embeddings",
        )
        self.energy_dropout = tf.keras.layers.Dropout(0.5)
        self.setup_inference_fn()

    def _build(self):
        """Dummy input for building model."""
        # fake inputs
        input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        duration_gts = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)
        f0_gts = tf.convert_to_tensor(
            [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]], tf.float32
        )
        energy_gts = tf.convert_to_tensor(
            [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]], tf.float32
        )
        self(
            input_ids=input_ids, speaker_ids=speaker_ids, duration_gts=duration_gts,
            f0_gts=f0_gts, energy_gts=energy_gts,
        )

    def call(
        self, input_ids, speaker_ids, duration_gts, f0_gts, energy_gts,
        training=False, **kwargs,
    ):
        """Call logic."""
        attention_mask = tf.math.not_equal(input_ids, 0)
        embedding_output = self.embeddings([input_ids, speaker_ids], training=training)
        encoder_output = self.encoder(
            [embedding_output, attention_mask], training=training
        )
        last_encoder_hidden_states = encoder_output[0]
        # var predictor
        duration_outputs = self.duration_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask]
        )  # [batch_size, length]
        f0_outputs = self.f0_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask], training=training
        )
        energy_outputs = self.energy_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask], training=training
        )
        f0_embedding = self.f0_embeddings(
            tf.expand_dims(f0_gts, 2)
        )  # [barch_size, mel_length, feature]
        energy_embedding = self.energy_embeddings(
            tf.expand_dims(energy_gts, 2)
        )  # [barch_size, mel_length, feature]
        # apply dropout both training/inference
        f0_embedding = self.f0_dropout(f0_embedding, training=True)
        energy_embedding = self.energy_dropout(energy_embedding, training=True)
        # sum features
        last_encoder_hidden_states += f0_embedding + energy_embedding
        # length regulator
        length_regulator_outputs, encoder_masks = self.length_regulator(
            [last_encoder_hidden_states, duration_gts], training=training
        )
        # create decoder positional embedding
        decoder_pos = tf.range(
            1, tf.shape(length_regulator_outputs)[1] + 1, dtype=tf.int32
        )
        masked_decoder_pos = tf.expand_dims(decoder_pos, 0) * encoder_masks
        # decoder
        decoder_output = self.decoder(
            [length_regulator_outputs, speaker_ids, encoder_masks, masked_decoder_pos],
            training=training,
        )
        last_decoder_hidden_states = decoder_output[0]
        # post processing, you can use not only last decoder_hidden_states by sum or concate
        mels_before = self.mel_dense(last_decoder_hidden_states)
        mels_after = (
            self.postnet([mels_before, encoder_masks], training=training) + mels_before
        )
        return (mels_before, mels_after, duration_outputs, f0_outputs, energy_outputs)

    def _inference(
        self, input_ids, speaker_ids, speed_ratios, f0_ratios, energy_ratios, **kwargs,
    ):
        """Call logic."""
        attention_mask = tf.math.not_equal(input_ids, 0)
        embedding_output = self.embeddings([input_ids, speaker_ids], training=False)
        encoder_output = self.encoder(
            [embedding_output, attention_mask], training=False
        )
        last_encoder_hidden_states = encoder_output[0]
        # expand ratios
        speed_ratios = tf.expand_dims(speed_ratios, 1)  # [B, 1]
        f0_ratios = tf.expand_dims(f0_ratios, 1)  # [B, 1]
        energy_ratios = tf.expand_dims(energy_ratios, 1)  # [B, 1]
        # var predictor: duration
        duration_outputs = self.duration_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask]
        )  # [batch_size, length]
        duration_outputs = tf.nn.relu(tf.math.exp(duration_outputs) - 1.0)
        duration_outputs = tf.cast(
            tf.math.round(duration_outputs * speed_ratios), tf.int32
        )
        # var predictor: pitch
        f0_outputs = self.f0_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask], training=False
        )
        f0_outputs *= f0_ratios
        # var predictor: energy
        energy_outputs = self.energy_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask], training=False
        )
        energy_outputs *= energy_ratios
        # var predictor: dropout???
        f0_embedding = self.f0_dropout(
            self.f0_embeddings(tf.expand_dims(f0_outputs, 2)), training=True
        )
        energy_embedding = self.energy_dropout(
            self.energy_embeddings(tf.expand_dims(energy_outputs, 2)), training=True
        )
        # sum features
        last_encoder_hidden_states += f0_embedding + energy_embedding
        # length regulator
        length_regulator_outputs, encoder_masks = self.length_regulator(
            [last_encoder_hidden_states, duration_outputs], training=False
        )
        # create decoder positional embedding
        decoder_pos = tf.range(
            1, tf.shape(length_regulator_outputs)[1] + 1, dtype=tf.int32
        )
        masked_decoder_pos = tf.expand_dims(decoder_pos, 0) * encoder_masks
        # decode
        decoder_output = self.decoder(
            [length_regulator_outputs, speaker_ids, encoder_masks, masked_decoder_pos],
            training=False,
        )
        last_decoder_hidden_states = decoder_output[0]
        # post processing, you can use not only last decoder_hidden_states by sum or concate
        mel_before = self.mel_dense(last_decoder_hidden_states)
        mel_after = (
            self.postnet([mel_before, encoder_masks], training=False) + mel_before
        )
        #return (mel_before, mel_after, duration_outputs, f0_outputs, energy_outputs)
        return mel_after

    def setup_inference_fn(self):
        self.inference = tf.function(
            self._inference,
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="input_ids"),
                tf.TensorSpec(shape=[None,], dtype=tf.int32, name="speaker_ids"),
                tf.TensorSpec(shape=[None,], dtype=tf.float32, name="speed_ratios"),
                tf.TensorSpec(shape=[None,], dtype=tf.float32, name="f0_ratios"),
                tf.TensorSpec(shape=[None,], dtype=tf.float32, name="energy_ratios"),
            ],
        )



