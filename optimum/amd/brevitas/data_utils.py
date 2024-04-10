# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import random
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm

from optimum.utils.normalized_config import NormalizedConfigManager
from transformers import AutoConfig


if TYPE_CHECKING:
    from .configuration import BrevitasQuantizationConfig

HIDDEN_SIZE_KEYS = ["d_model", "hidden_size"]
NUM_HEADS_KEYS = ["num_attention_heads"]


@torch.no_grad()
def recursive_to_device(tensor_or_iterable: Union[Iterable, torch.Tensor], device) -> None:
    if isinstance(tensor_or_iterable, torch.Tensor):
        return tensor_or_iterable.to(device)
    elif isinstance(tensor_or_iterable, tuple):  # Special handling of tuples, since they are immutable
        tmp_list = []
        for i in tensor_or_iterable:
            tmp_list.append(recursive_to_device(i, device))
        return tuple(tmp_list)
    elif isinstance(tensor_or_iterable, Iterable):
        for i in tensor_or_iterable:
            tensor_or_iterable[i] = recursive_to_device(i, device)
        return tensor_or_iterable
    else:
        raise ValueError(f"Cannot move {type(tensor_or_iterable)} to {device}")


@torch.no_grad()
def compute_perplexity(
    model: torch.nn.Module,
    data: List[Dict],
    context_length: int,
    tokenizer: Any,
    seed: int = 0,
    add_bos_token_id: bool = True,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    model = model.eval()

    cross_entropy_loss = nn.CrossEntropyLoss()

    nlls = []
    total_eval_length = 0
    for sample in tqdm(data, desc="Computing perplexity..."):
        batch_size, sample_length = sample["input_ids"].shape
        for start_index in range(0, sample_length, context_length):
            end_index = min(start_index + 2 * context_length, sample_length - 1)

            eval_length = end_index - start_index + 1 - context_length
            if eval_length <= 0:
                continue

            subsample = {
                "input_ids": sample["input_ids"][:, start_index : end_index + 1],
                "attention_mask": sample["attention_mask"][:, start_index : end_index + 1],
            }

            # In case we are using torch.fx, we can not have optional inputs, and we have traced the model with past_key_values inputs, thus we need them here as well.
            if "past_key_values" in sample and isinstance(model, torch.fx.GraphModule):
                subsample["past_key_values"] = sample["past_key_values"]

            dtype = subsample["input_ids"].dtype
            device = subsample["input_ids"].device

            # Add BOS token.
            current_context_length = context_length
            if add_bos_token_id and not torch.all(subsample["input_ids"][:, 0] == tokenizer.bos_token_id):
                bos_tensor = torch.full((batch_size, 1), fill_value=tokenizer.bos_token_id, dtype=dtype, device=device)
                bos_mask_tensor = torch.full((batch_size, 1), fill_value=1, dtype=dtype, device=device)

                subsample["input_ids"] = torch.cat((subsample["input_ids"], bos_tensor), dim=-1)
                subsample["attention_mask"] = torch.cat((subsample["attention_mask"], bos_mask_tensor), dim=-1)

                current_context_length += 1

            use_accelerate = hasattr(model, "hf_device_map")
            if not use_accelerate or (use_accelerate and not hasattr(model, "_hf_hook")):
                device = next(model.parameters()).device
                for name, val in subsample.items():
                    subsample[name] = recursive_to_device(val, device)
            else:
                # In accelerate by default `io_same_device=True`, and here we want the of the model output on device.
                device = model._hf_hook.execution_device
                for name, val in subsample.items():
                    subsample[name] = recursive_to_device(val, device)

            lm_logits = model(**subsample)["logits"]

            reference_labels = subsample["input_ids"][:, current_context_length:]

            shift_logits = lm_logits[:, current_context_length - 1 : -1]

            # Fuse batch and sequence length dimensions.
            reference_labels = reference_labels.view(reference_labels.shape[-1])
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])

            loss = cross_entropy_loss(shift_logits, reference_labels)
            neg_log_likelihood = loss.float() * eval_length

            total_eval_length += eval_length
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / total_eval_length)

    return ppl


def get_wikitext2(
    tokenizer: Any, seqlen: int, nsamples: int, split: str = "train", fuse_sequences: bool = True, seed: int = 42
):
    random.seed(seed)

    if split == "train":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif split == "validation":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    if fuse_sequences:
        data = data.shuffle(seed=seed)
        # wikitext2 is too big.
        tokenized_data = tokenizer("\n\n".join(data["text"])[:100000], return_tensors="pt")

        dataset = []
        for _ in range(nsamples):
            i = random.randint(0, tokenized_data.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = tokenized_data.input_ids[:, i:j]
            attention_mask = torch.ones((1, seqlen), dtype=torch.int64)
            dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    else:
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
                input_ids = enc["input_ids"][:, start_idx : end_idx + 1]

                # Add BOS token.
                input_ids[:, 0] = tokenizer.bos_token_id

                dataset.append({"input_ids": input_ids, "attention_mask": attention_mask})
                pbar.update(1)

    return dataset


def get_c4(
    tokenizer: Any, seqlen: int, nsamples: int, split: str = "train", fuse_sequences: bool = True, seed: int = 42
):
    random.seed(seed)

    if split == "train":
        data = load_dataset("allenai/c4", split="train", data_files={"train": "en/c4-train.00000-of-01024.json.gz"})
    elif split == "validation":
        data = load_dataset(
            "allenai/c4",
            split="validation",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        )

    if fuse_sequences:
        data = data.shuffle(seed=seed)[:10000]  # c4 is too big.
        full_text = "\n\n".join(data["text"])
        tokenized_data = tokenizer(full_text, return_tensors="pt")

        dataset = []
        for _ in range(nsamples):
            i = random.randint(0, tokenized_data.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = tokenized_data.input_ids[:, i:j]
            attention_mask = torch.ones((1, seqlen), dtype=torch.int64)
            dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    else:
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
                input_ids = enc["input_ids"][:, start_idx : end_idx + 1]

                # Add BOS token.
                input_ids[:, 0] = tokenizer.bos_token_id

                dataset.append({"input_ids": input_ids, "attention_mask": attention_mask})
                pbar.update(1)

    return dataset


class DatasetToDevice(torch.utils.data.Dataset):
    def __init__(self, data: List, device: Optional[Union[str, torch.device]]):
        super().__init__()
        self.data = data
        self.device = device

    def __getitem__(self, idx):
        if self.device is not None:
            return {name: recursive_to_device(val, self.device) for name, val in self.data[idx].items()}
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


def get_dataset_for_model(
    model_name_or_path: str,
    qconfig: "BrevitasQuantizationConfig",
    dataset_name: str,
    tokenizer: Any,
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
    split: str = "train",
    fuse_sequences: bool = True,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Get a dataset.

    Args:
        model_name_or_path (`str`):
            A local folder containing the model or the model hosted on the Hugging Face Hub.
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

    data = get_dataset_fn(
        tokenizer=tokenizer, nsamples=nsamples, seqlen=seqlen, split=split, fuse_sequences=fuse_sequences, seed=seed
    )

    # In case the dataset is loaded to be used with an fx.GraphModule, we need to add empty past_key_values inputs in the dataset.
    if qconfig.requires_fx_graph():
        config = AutoConfig.from_pretrained(model_name_or_path)

        normalized_config_class = NormalizedConfigManager.get_normalized_config_class(config.model_type)
        normalized_config = normalized_config_class(config)

        num_heads = normalized_config.num_attention_heads
        head_dim = normalized_config.hidden_size // num_heads
        num_layers = normalized_config.num_layers

        for sample in data:
            sample["past_key_values"] = tuple(
                (
                    torch.zeros(1, num_heads, 0, head_dim, device=sample["input_ids"].device),
                    torch.zeros(1, num_heads, 0, head_dim, device=sample["input_ids"].device),
                )
                for _ in range(num_layers)
            )

    data = DatasetToDevice(data, device=device)

    return data
