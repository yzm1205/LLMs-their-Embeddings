import time
from pathlib import Path
from typing import Tuple, Union, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import json
from utils import full_path

with open("./key.json", "r", encoding="utf-8") as f:
    keys = json.load(f)

hf_auth = keys["hf_llama_2"]


def get_llama_model_and_tokenizer(
    model_pt: Path,
    hf_token: str,
    device: str = "cuda",
    model_cls: Union[AutoModel, AutoModelForCausalLM] = AutoModel,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_pt, token=hf_token)

    # This is required because LLAMA2 does not have PAD token
    tokenizer.pad_token = tokenizer.eos_token

    model = model_cls.from_pretrained(
        model_pt,
        device_map=device,
        torch_dtype=torch.float16,
    )
    print(f"Time taken to load model and tokenizer is {time.time() - start}")
    return model.eval(), tokenizer


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def llama2(
    sent: Union[str, List[str]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str = "cuda",
    auto_model: bool = True,
    max_length: int = 1024,
) -> torch.Tensor:
    start = time.time()
    device = torch.device(device)
    with torch.no_grad():
        sentence_encode = tokenizer(
            sent,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        if auto_model:
            sentence_embedding = model(**sentence_encode)
            sentence_embedding = sentence_embedding.last_hidden_state.detach().cpu()
        else:  # CausalLM
            sentence_embedding = model(**sentence_encode, output_hidden_states=True)
            sentence_embedding = sentence_embedding.hidden_states[-1].detach().cpu()
    sentence_embedding = mean_pooling(
        sentence_embedding, sentence_encode["attention_mask"].cpu()
    )
    # print(
    #     f"Time taken to compute embeddings is {time.time() - start}, Shape: {sentence_embedding.shape}"
    # )
    return sentence_embedding


def main():
    hf_token = hf_auth
    sent = [
        "I go to school",
        "Unequal length sentences, that's why I have a long sentence here",
        "Unequal length sentences, that's why I have a long sentence here. Unequal length sentences, that's why I have a long sentence here",
    ]
    batch = 1000
    sent_batch = sent * batch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = full_path(f"/data/naman/sharded_llama2/llama2_shard_size_1GB")
    model, tokenizer = get_llama_model_and_tokenizer(model_dir, hf_token, device=device)

    for _ in range(10):
        sentence_embeddings = llama2(sent_batch, model, tokenizer, device=device)
    print(sentence_embeddings)
    print(sentence_embeddings.numpy())
    print(sentence_embeddings.shape)


if __name__ == "__main__":
    main()
    print("Done!")
