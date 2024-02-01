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
def model_eval_accelerate(model, data: List[Dict], context_length: int, tokenizer):    
    model = model.eval()
    device = data[0]["input_ids"].device

    ppls = []
    for sample in tqdm(data):
        sample_length = sample["input_ids"].shape[1]
        for start_index in range(0, sample_length, context_length * 2):
            end_index = min(start_index + sample_length, sample_length - 1)

            subsample = {"input_ids": sample["input_ids"][:, start_index:end_index + 1], "attention_mask": sample["attention_mask"][:, start_index:end_index + 1]}

            # Add BOS token.
            subsample["input_ids"][0] = tokenizer.bos_token_id

            with torch.no_grad():
                lm_logits = model(**subsample)["logits"]

            reference_labels = subsample["input_ids"][:, context_length:]

            shift_logits = lm_logits[:, context_length - 1:-1]

            # Fuse batch and sequence length dimensions.
            reference_labels = reference_labels.view(reference_labels.shape[-1])
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])

            loss_fct = nn.CrossEntropyLoss(reduction="none")

            loss = loss_fct(shift_logits, reference_labels)

            ppls_subsample = torch.exp(loss)

            ppls += ppls_subsample.tolist()
    
    return np.mean(ppls)


def get_wikitext2(tokenizer: Any, seqlen: int, nsamples: int, split: str = "train"):
    if split == "train":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif split == "validation":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
    dataset = []
    with tqdm(total=nsamples) as pbar:
        while len(dataset) < nsamples:
            data_index = random.randint(0, len(data) - 1)

            enc = tokenizer(data[data_index]["text"], return_tensors="pt")

            if enc["input_ids"].shape[1] < seqlen:
                continue

            start_idx = random.randint(0, enc["input_ids"].shape[1] - seqlen)
            end_idx = start_idx + seqlen - 1
            attention_mask = torch.ones((1, seqlen), dtype=torch.int64)
            dataset.append({"input_ids": enc["input_ids"][:, start_idx:end_idx + 1], "attention_mask": attention_mask})
            pbar.update(1)
    
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
    with tqdm(total=nsamples) as pbar:
        while len(dataset) < nsamples:
            data_index = random.randint(0, len(data) - 1)

            enc = tokenizer(data[data_index]["text"], return_tensors="pt")

            if enc["input_ids"].shape[1] < seqlen:
                continue

            start_idx = random.randint(0, enc["input_ids"].shape[1] - seqlen)
            end_idx = start_idx + seqlen - 1
            attention_mask = torch.ones((1, seqlen), dtype=torch.int64)
            dataset.append({"input_ids": enc["input_ids"][:, start_idx:end_idx + 1], "attention_mask": attention_mask})
            pbar.update(1)
    
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
