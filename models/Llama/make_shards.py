import time
from pathlib import Path
from typing import Union

from transformers import AutoTokenizer, AutoModel


from utils import mkdir_p


def shard_llama_model_and_tokenizer(
    parent_save_dir: Path,
    token: str,
    max_shard_size: Union[str, int] = "3GB",
    model_name: str = "meta-llama/Llama-2-7b-hf",
) -> None:
    """
    :param parent_save_dir: Parent directory under which model shards dir will be saved
    :param token: huggingface token to access the llama weights
    :param max_shard_size: shard size. int means number of bytes
    :param model_name: HF Model Name
    :return:
    """
    start = time.time()
    save_directory = parent_save_dir.joinpath(f"llama2_shard_size_{max_shard_size}")
    print(f"Shard Dir: {str(save_directory)}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModel.from_pretrained(model_name, token=token)

    model.save_pretrained(save_directory, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(save_directory)
    print(f"Time Taken: {time.time() - start}")


def main():
    token = "" # llm token
    shard_size = "512MB"
    parent_save_dir = mkdir_p("./data//sharded_llama2/")
    shard_llama_model_and_tokenizer(parent_save_dir, token, shard_size)


if __name__ == "__main__":
    main()
    print(f"Done")
