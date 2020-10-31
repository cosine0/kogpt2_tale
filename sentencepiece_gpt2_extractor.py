from argparse import ArgumentParser
from json import dump
from logging import basicConfig, getLogger
from os import linesep, remove
from os.path import exists
from tempfile import NamedTemporaryFile
from typing import Dict, List, Tuple

from requests import get
from sentencepiece import SentencePieceProcessor
from tqdm import trange, tqdm
from transformers import GPT2Tokenizer

basicConfig()
logger = getLogger()


class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models.
    https://github.com/google/sentencepiece
    """

    def __init__(self, model: str):
        # Get SentencePiece
        self.sp = SentencePieceProcessor()
        self.sp.Load(model)
        self.gpt2_tokenizer_example = GPT2Tokenizer.from_pretrained('gpt2')

    def to_gpt2_token(self, original_token):
        return "".join(self.gpt2_tokenizer_example.byte_encoder[b] for b in original_token.encode("utf-8"))

    def extract(self) -> Tuple[Dict[str, int], List[Tuple]]:
        sp = self.sp
        vocab = {self.to_gpt2_token(sp.id_to_piece(index)): index for index in trange(sp.GetPieceSize())}

        # Merges
        merges = []
        for piece_l in tqdm(vocab.keys(), total=sp.GetPieceSize()):
            for piece_r in vocab.keys():
                merge = f"{piece_l}{piece_r}"
                piece_id = vocab.get(merge, None)

                if piece_id:
                    merges += [(piece_l, piece_r, piece_id)]
        merges = sorted(merges, key=lambda val: val[2])
        merges = [(val[0], val[1]) for val in merges]

        return vocab, merges


if __name__ == "__main__":
    parser = ArgumentParser("SentencePiece vocab extractor")
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["sentencepiece", "youtokentome"],
        help="Indicate the format of the file.",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="SentencePiece model to extract vocab from."
    )
    parser.add_argument(
        "--vocab-output-path",
        type=str,
        required=True,
        help="Path where the vocab.json file will be extracted",
    )
    parser.add_argument(
        "--merges-output-path",
        type=str,
        required=True,
        help="Path where the merges file will be extracted",
    )

    # Parse cli arguments
    args = parser.parse_args()

    try:
        if args.model.startswith("http"):
            # Saving model
            with NamedTemporaryFile("wb", delete=False) as f:
                logger.info("Writing content from {} to {}".format(args.model, f.name))
                response = get(args.model, allow_redirects=True)
                f.write(response.content)

                args.remote_model = args.model
                args.model = f.name

        # Allocate extractor
        extractor = SentencePieceExtractor(args.model)

        logger.info(f"Using {type(extractor).__name__}")

        # Open output files and let's extract model information
        with open(args.vocab_output_path, "w") as vocab_f:
            with open(args.merges_output_path, "w") as merges_f:
                # Do the extraction
                vocab, merges = extractor.extract()

                # Save content
                dump(vocab, vocab_f)
                merges_f.writelines(map(lambda x: f"{x[0]} {x[1]}{linesep}", merges))
    finally:
        # If model was downloaded from internet we need to cleanup the tmp folder.
        if hasattr(args, "remote_model") and exists(args.model):
            remove(args.model)
