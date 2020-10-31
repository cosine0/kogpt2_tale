from typing import List, Optional

from transformers.tokenization_t5 import T5Tokenizer


class TaleTokenizer(T5Tokenizer):
    def __init__(self, spiece_filename):
        super().__init__(
            spiece_filename,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<pad>",
            pad_token="<pad>",
            add_prefix_space=False,
        )

    def _tokenize(self, text, sample=False):
