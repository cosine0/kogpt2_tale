from typing import Optional

import torch
from transformers import T5Tokenizer, GPT2LMHeadModel, PretrainedConfig, PreTrainedTokenizer, TextDataset, Trainer, \
    DataCollatorForLanguageModeling, TrainingArguments

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_dataset(
        data_file_path,
        tokenizer: PreTrainedTokenizer,
        max_len,
        cache_dir: Optional[str] = None,
):
    def _dataset(file_path):
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=max_len,
            cache_dir=cache_dir,
        )

    return _dataset(data_file_path)


def main():
    tokenizer = T5Tokenizer(
        'kogpt2_news_wiki_ko_cased_818bfa919d.spiece',
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<pad>",
        pad_token="<pad>",
        add_prefix_space=False,
    )

    model = GPT2LMHeadModel(PretrainedConfig.from_json_file("config.json"))
    model.load_state_dict(torch.load('pytorch_kogpt2_676e9bcfa7.params'), strict=False)
    model.to(device)

    train_dataset = get_dataset('data/tale.txt', tokenizer, max_len=model.config.max_length, cache_dir='data')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    args = TrainingArguments(output_dir='output', num_train_epochs=200, per_device_train_batch_size=128,
                             prediction_loss_only=True, dataloader_drop_last=True)
    trainer = Trainer(
        args=args,
        model=model,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model()

    # input_ids = tokenizer.encode("안녕", add_special_tokens=False, return_tensors="pt").to(device)
    # output_sequences = model.generate(input_ids=input_ids, do_sample=True, max_length=100, num_return_sequences=3,
    #                                   num_beams=4, min_length=10)
    # for generated_sequence in output_sequences:
    #     generated_sequence = generated_sequence.tolist()
    #     print(
    #         "GENERATED SEQUENCE : {0}".format(tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)))


if __name__ == '__main__':
    main()
