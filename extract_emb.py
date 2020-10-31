import torch
from transformers import XLNetTokenizer, GPT2LMHeadModel, PretrainedConfig, AutoModel

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

    model = GPT2LMHeadModel(PretrainedConfig.from_json_file("config.json"))
    model.load_state_dict(torch.load('pytorch_kogpt2_676e9bcfa7.params'), strict=False)
    emb = model.get_input_embeddings()
    emb.weight.detach().numpy().tofile('kogpt2_emb.npy')

    model = AutoModel.from_pretrained('xlnet-base-cased')
    emb = model.get_input_embeddings()
    emb.weight.detach().numpy().tofile('xlnet_emb.npy')


if __name__ == '__main__':
    main()
