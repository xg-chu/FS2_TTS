import json
import tensorflow as tf
import soundfile as sf

from tensorflow_tts.configs import FastSpeech2_Config, MBMelGANGenerator_Config
from tensorflow_tts.models import FastSpeech2, MBMelGANGenerator
from tensorflow_tts.text_processor import BakerProcessor

class Synthesizer():
    def __init__(self):
        self.fastspeech2_model_path = 'tensorflow_tts/weights/fastspeech2-200k.h5'
        self.melgan_model_path = 'tensorflow_tts/weights/mb.melgan-920k.h5'
        self.fastspeech2_config = FastSpeech2_Config()
        self.mb_melgan_config = MBMelGANGenerator_Config()
        self.processor = BakerProcessor()

    def load_model(self):
        self.fastspeech2 = FastSpeech2(config=self.fastspeech2_config, name="fastspeech2")
        self.fastspeech2._build()
        self.fastspeech2.load_weights(self.fastspeech2_model_path)
        self.mb_melgan = MBMelGANGenerator(config=self.mb_melgan_config, name="mb_melgan")
        self.mb_melgan._build()
        self.mb_melgan.load_weights(self.melgan_model_path)

    def synthesis(self, input_text, output_path):
        input_ids = self.processor.text_to_sequence(input_text, is_text=True)
        mel_outputs = self.fastspeech2.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        )
        audio = self.mb_melgan.inference(mel_outputs)[0, :-1, 0]
        audio = audio.numpy()
        sf.write(output_path, audio, 24000, "PCM_16")

if __name__ == "__main__":
    model = Synthesizer()
    model.load_model()
    model.synthesis("前方危险，立即回转", 'new.wav')
