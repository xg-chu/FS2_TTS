"""Perform preprocessing and raw feature extraction for Baker dataset."""
import os
import re
import soundfile as sf
from pypinyin import Style
from pypinyin.core import Pinyin
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from tensorflow_tts.text_processor.phoneme_mapper import BAKER_DICT

zh_pattern = re.compile("[\u4e00-\u9fa5]")
def is_zh(word):
    match = zh_pattern.search(word)
    return match is not None

class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass

class BakerProcessor():
    def __init__(self):
        self.pinyin_parser = Pinyin(MyConverter()).pinyin
        self.pinyin_dict = BAKER_DICT['pinyin_dict']
        self.all_phoneme = self.pinyin_dict.keys()
        self.phoneme_to_id = BAKER_DICT["symbol_to_id"]
        self.pause_list = [
            ',', '.', '!', ';', '(', ')', 
            '、', '！', '。', '，', '；', '—', '（', '）']
        # with open(loaded_path, "r") as f:
        #     data = json.load(f)
        # self.speakers_map = data["speakers_map"]
        # 
        # self.id_to_symbol = {int(k): v for k, v in data["id_to_symbol"].items()}

    def get_phoneme_from_char_and_pinyin(self, texts, pinyin):
        result = ["sil"]
        i = 0; j = 0
        while i < len(texts):
            if texts[i] == '#2' or texts[i] == '#3':
                result += [texts[i]]
                i += 1
            else:
                phoneme = pinyin[j][:-1]
                tone = pinyin[j][-1]
                if phoneme not in self.all_phoneme:
                    phoneme = pinyin[j][:-2]
                    p1, p2 = self.pinyin_dict[phoneme]
                    result += [p1, p2 + tone, "er5", "#0"]
                    i += 2; j += 1
                else:
                    p1, p2 = self.pinyin_dict[phoneme]
                    result += [p1, p2 + tone, "#0"]
                    i += 1; j += 1
        result[-1] = "sil"
        assert j == len(pinyin)
        return result

    def text_to_sequence(self, inputs, is_text=False):
        phonemes = inputs
        if is_text:
            #inputs = "".join([char for char in inputs if is_zh(char)])
            inputs_list = []
            for char in inputs:
                if is_zh(char):
                    inputs_list.append(char)
                elif char in self.pause_list:
                    inputs_list.append('#3')
            pinyin = self.pinyin_parser(inputs_list, style=Style.TONE3, errors="ignore")
            print(pinyin)
            new_pinyin = ["".join(x) for x in pinyin if '#' not in x]
            phonemes = self.get_phoneme_from_char_and_pinyin(inputs_list, new_pinyin)
        sequence = [self.phoneme_to_id[phoneme] for phoneme in phonemes]
        sequence += [self.phoneme_to_id['eos']]
        return sequence
