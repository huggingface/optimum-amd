# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import unittest

from parameterized import parameterized

from optimum.amd.brevitas import BrevitasQuantizationConfig, get_dataset_for_model
from transformers import AutoTokenizer


class TestDataLoading(unittest.TestCase):
    @parameterized.expand([("c4",), ("wikitext2")])
    def test_data_loading(self, dataset_name: str):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        qconfig = BrevitasQuantizationConfig()
        dataset = get_dataset_for_model(
            "gpt2",
            tokenizer=tokenizer,
            qconfig=qconfig,
            dataset_name=dataset_name,
            seqlen=128,
            nsamples=32,
            fuse_sequences=False,
        )

        self.assertTrue(len(dataset) == 32)
        self.assertTrue(dataset[0]["input_ids"].shape[1] == 128)

        qconfig = BrevitasQuantizationConfig()
        dataset = get_dataset_for_model(
            "gpt2",
            tokenizer=tokenizer,
            qconfig=qconfig,
            dataset_name=dataset_name,
            seqlen=2048,
            nsamples=256,
            fuse_sequences=True,
        )

        self.assertTrue(len(dataset) == 256)
        self.assertTrue(dataset[0]["input_ids"].shape[1] == 2048)
