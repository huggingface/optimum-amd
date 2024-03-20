import os
import unittest

import torch
import zentorch  # noqa: F401
from parameterized import parameterized  # type: ignore
from testing_utils import (
    IMAGE_DIFFUSION_KWARGS,
    SUPPORTED_MODELS_TINY,
    SUPPORTED_MODELS_TINY_IMAGE_DIFFUSION,
    SUPPORTED_MODELS_TINY_TEXT_GENERATION,
    TEXT_GENERATION_KWARGS,
    get_model_or_pipe_inputs,
    load_and_compile_model,
    load_and_compile_pipeline,
    load_model_or_pipe,
)


# set to 1 to run the model in eager mode
EAGER_DEBUG = os.environ.get("EAGER_DEBUG", "0") == "1"

CPU_COUNT = os.cpu_count()
LD_PRELOAD = "/usr/lib/x86_64-linux-gnu/libjemalloc.so"
MALLOC_CONF = "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
os.environ["OMP_NUM_THREADS "] = f"{CPU_COUNT}"
os.environ["OMP_DYNAMIC"] = "False"
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
os.environ["ZENDNN_GEMM_ALGO"] = "4"
os.environ["GOMP_CPU_AFFINITY"] = f"0-{CPU_COUNT - 1}"
os.environ["LD_PRELOAD"] = LD_PRELOAD
os.environ["MALLOC_CONF"] = MALLOC_CONF


class TestZenTorchPlugin(unittest.TestCase):
    @parameterized.expand(SUPPORTED_MODELS_TINY.keys())
    def test_text_classification_model(self, model_type: str):
        model_id_and_tasks = SUPPORTED_MODELS_TINY[model_type]

        for model_id, tasks in model_id_and_tasks.items():
            for task in tasks:
                inputs = get_model_or_pipe_inputs(model_id, task)

                if EAGER_DEBUG:
                    model = load_model_or_pipe(model_id, task)
                    _ = model(**inputs).logits
                    continue

                inductor_model = load_and_compile_model(model_id, task, backend="inductor")
                inductor_logits = inductor_model(**inputs).logits

                zentorch_model = load_and_compile_model(model_id, task, backend="zentorch")
                zentorch_logits = zentorch_model(**inputs).logits

                torch.testing.assert_close(inductor_logits, zentorch_logits, rtol=1e-3, atol=1e-5)

    @parameterized.expand(SUPPORTED_MODELS_TINY_TEXT_GENERATION.keys())
    def test_text_generation_model(self, model_type: str):
        model_id_and_tasks = SUPPORTED_MODELS_TINY_TEXT_GENERATION[model_type]

        for model_id, tasks in model_id_and_tasks.items():
            for task in tasks:
                inputs = get_model_or_pipe_inputs(model_id, task)

                if EAGER_DEBUG:
                    model = load_model_or_pipe(model_id, task)
                    _ = model.generate(**inputs, **TEXT_GENERATION_KWARGS).scores
                    continue

                inductor_model = load_and_compile_model(model_id, task, backend="inductor")
                # use logits in transformers next release
                # https://github.com/huggingface/transformers/issues/14498#issuecomment-1953014058
                inductor_scores = inductor_model.generate(**inputs, **TEXT_GENERATION_KWARGS).scores
                inductor_scores = torch.concat(inductor_scores)

                zentorch_model = load_and_compile_model(model_id, task, backend="zentorch")
                # use logits in transformers next release
                # https://github.com/huggingface/transformers/issues/14498#issuecomment-1953014058
                zentorch_scores = zentorch_model.generate(**inputs, **TEXT_GENERATION_KWARGS).scores
                zentorch_scores = torch.concat(zentorch_scores)

                torch.testing.assert_close(inductor_scores, zentorch_scores, rtol=1e-3, atol=1e-5)

    @parameterized.expand(SUPPORTED_MODELS_TINY_IMAGE_DIFFUSION.keys())
    def test_image_diffusion_pipe(self, model_type: str):
        model_id_and_tasks = SUPPORTED_MODELS_TINY_IMAGE_DIFFUSION[model_type]

        for model_id, tasks in model_id_and_tasks.items():
            for task in tasks:
                inputs = get_model_or_pipe_inputs(model_id, task)

                if EAGER_DEBUG:
                    pipe = load_model_or_pipe(model_id, task)
                    _ = pipe(**inputs, **IMAGE_DIFFUSION_KWARGS).images
                    continue

                # we get a new instance of the pipe for every backend to avoid any side effects
                inductor_pipe = load_and_compile_pipeline(model_id, task, backend="inductor")
                inductor_images = inductor_pipe(**inputs, **IMAGE_DIFFUSION_KWARGS).images

                zentorch_pipe = load_and_compile_pipeline(model_id, task, backend="zentorch")
                zentorch_images = zentorch_pipe(**inputs, **IMAGE_DIFFUSION_KWARGS).images

                torch.testing.assert_close(inductor_images, zentorch_images, rtol=1e-3, atol=1e-5)
