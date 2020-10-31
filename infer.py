import torch
from transformers import XLNetTokenizer, GPT2LMHeadModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    tokenizer = XLNetTokenizer(
        'kogpt2_news_wiki_ko_cased_818bfa919d.spiece',
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<pad>",
        pad_token="<pad>",
        add_prefix_space=False,
    )

    model = GPT2LMHeadModel.from_pretrained('output')
    model.to(device)

    input_ids = tokenizer.encode("쥐를 친구로 둔 어떤 고양이가 살고 있었어요.", add_special_tokens=False, return_tensors="pt").to(device)
    output_sequences = model.generate(input_ids, do_sample=True, max_length=100, num_return_sequences=3,
                                      num_beams=16, min_length=10, repetition_penalty=10.)
    for generated_sequence in output_sequences:
        generated_sequence = generated_sequence.tolist()
        print(
            "GENERATED SEQUENCE : {0}".format(tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)))


if __name__ == '__main__':
    main()
