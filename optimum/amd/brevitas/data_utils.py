# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import random
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm


@torch.no_grad()
def model_eval_accelerate(model, data: List[Dict], seqlen: int):
    encodings = {"input_ids": torch.empty((0,), dtype=torch.int64)}

    # TODO: It is bad practice to compute perplexity like that (mixing up sequences in a single long sequence). The way evaluate handles it is cleaner: https://github.com/huggingface/evaluate/blob/8dfe05784099fb9af55b8e77793205a3b7c86465/metrics/perplexity/perplexity.py#L140
    for batch in data:
        encodings["input_ids"] = torch.cat((encodings["input_ids"], batch["input_ids"].squeeze(0)), dim=0)

    encodings["input_ids"] = encodings["input_ids"].unsqueeze(0)

    nsamples = encodings["input_ids"].numel() // seqlen
    use_cache = model.config.use_cache
    model.config.use_cache = False
    with torch.no_grad():
        nlls = []
        for i in tqdm(range(nsamples)):
            batch = encodings["input_ids"][:, (i * seqlen) : ((i + 1) * seqlen)].cuda()
            attention_mask = torch.ones_like(batch)
            lm_logits = model(input_ids=batch, attention_mask=attention_mask)["logits"]
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = (encodings["input_ids"][:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]).cuda()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    model.config.use_cache = use_cache
    return ppl


def get_wikitext2(tokenizer: Any, seqlen: int, nsamples: int, split: str = "train"):
    if split == "train":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif split == "validation":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # length of 288059 should be enough
    text = "".join([" \n" if s == "" else s for s in data["text"][:1000]])

    enc = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return dataset


def get_c4(tokenizer: Any, seqlen: int, nsamples: int, split: str = "train"):
    if split == "train":
        data = load_dataset("allenai/c4", split="train", data_files={"train": "en/c4-train.00000-of-01024.json.gz"})
    elif split == "validation":
        data = load_dataset(
            "allenai/c4",
            split="validation",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        )
    dataset = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(data) - 1)
            enc = tokenizer(data[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})

    return dataset


def get_dataset(
    dataset_name: str, tokenizer: Any, nsamples: int = 128, seqlen: int = 2048, seed: int = 0, split: str = "train"
):
    """
    Get the dataset from the original paper of GPTQ

    Args:
        dataset_name (`str`):
            Dataset name. Available options are `['wikitext2', 'c4']`.
        tokenizer (`Any`):
            Tokenizer of the model
        nsamples (`int`, defaults to `128`):
            Number of samples
        seqlen (`int`, defaults to `2048`):
            The sequence length of the model
        seed (`int`, defaults to `0`):
            Seed
        split (`str`, defaults to `train`):
            Split of the dataset. Can be either "train" or "validation"
    Returns:
        `List[Dict[str,torch.LongTensor]]`: The tokenized dataset.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    get_dataset_map = {
        "wikitext2": get_wikitext2,
        "c4": get_c4,
    }
    if split not in ["train", "validation"]:
        raise ValueError(f"The split need to be 'train' or 'validation' but found {split}")
    if dataset_name not in get_dataset_map:
        raise ValueError(f"Expected a value in {list(get_dataset_map.keys())} but found {dataset_name}")
    get_dataset_fn = get_dataset_map[dataset_name]
    return get_dataset_fn(tokenizer=tokenizer, nsamples=nsamples, seqlen=seqlen, split=split)
