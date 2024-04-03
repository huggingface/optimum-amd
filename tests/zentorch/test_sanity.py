import os

import pytest
import torch
import zentorch  # noqa: F401
from testing_utils import (
    FAST_DIFFUSION_KWARGS,
    FAST_TEXT_GENERATION_KWARGS,
    SUPPORTED_DIFFUSION_PIPELINES_TINY,
    SUPPORTED_SIMPLE_MODELS_TINY,
    SUPPORTED_TEXT_GENERATION_MODELS_TINY,
    compile_diffusion_pipeline,
    compile_transformers_model,
    get_diffusion_pipeline_inputs,
    get_transformers_model_inputs,
    load_diffusion_pipeline,
    load_transformers_model,
)


# to avoid fast tokenizers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"


def test_amdrun_zentorch_setup():
    assert os.environ["OMP_DYNAMIC"] == "False"
    assert os.environ["ZENDNN_GEMM_ALGO"] == "4"
    assert os.environ["OMP_WAIT_POLICY"] == "ACTIVE"
    assert os.environ["OMP_NUM_THREADS"] == f"{os.cpu_count()}"
    assert os.environ["GOMP_CPU_AFFINITY"] == f"0-{os.cpu_count() - 1}"
    assert os.environ["LD_PRELOAD"] == "/usr/lib/x86_64-linux-gnu/libjemalloc.so"
    assert (
        os.environ["MALLOC_CONF"]
        == "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
    )


@pytest.mark.parametrize("model_type", SUPPORTED_SIMPLE_MODELS_TINY.keys())
def test_simple_models(model_type: str):
    model_id_and_tasks = SUPPORTED_SIMPLE_MODELS_TINY[model_type]

    for model_id, tasks in model_id_and_tasks.items():
        for task in tasks:
            inputs = get_transformers_model_inputs(model_id, task)

            model = load_transformers_model(model_id, task)
            inductor_model = compile_transformers_model(model, backend="inductor")
            inductor_logits = inductor_model.forward(**inputs).logits

            model = load_transformers_model(model_id, task)
            zentorch_model = compile_transformers_model(model, backend="zentorch")
            zentorch_logits = zentorch_model.forward(**inputs).logits

            torch.testing.assert_close(inductor_logits, zentorch_logits, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("model_type", SUPPORTED_TEXT_GENERATION_MODELS_TINY.keys())
def test_text_generation_models(model_type: str):
    model_id_and_tasks = SUPPORTED_TEXT_GENERATION_MODELS_TINY[model_type]

    for model_id, tasks in model_id_and_tasks.items():
        for task in tasks:
            inputs = get_transformers_model_inputs(model_id, task)

            model = load_transformers_model(model_id, task)
            inductor_model = compile_transformers_model(model, backend="inductor")
            inductor_logits = inductor_model.generate(**inputs, **FAST_TEXT_GENERATION_KWARGS).logits
            inductor_logits = torch.stack(inductor_logits, dim=1)

            model = load_transformers_model(model_id, task)
            zentorch_model = compile_transformers_model(model, backend="zentorch")
            zentorch_logits = zentorch_model.generate(**inputs, **FAST_TEXT_GENERATION_KWARGS).logits
            zentorch_logits = torch.stack(zentorch_logits, dim=1)

            torch.testing.assert_close(inductor_logits, zentorch_logits, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("pipeline_type", SUPPORTED_DIFFUSION_PIPELINES_TINY.keys())
def test_diffusion_pipelines(pipeline_type: str):
    pipeline_id_and_tasks = SUPPORTED_DIFFUSION_PIPELINES_TINY[pipeline_type]

    for pipeline_id, tasks in pipeline_id_and_tasks.items():
        for task in tasks:
            inputs = get_diffusion_pipeline_inputs(pipeline_id, task)

            pipeline = load_diffusion_pipeline(pipeline_id, task)
            inductor_pipeline = compile_diffusion_pipeline(pipeline, backend="inductor")
            inductor_images = inductor_pipeline(**inputs, **FAST_DIFFUSION_KWARGS).images

            pipeline = load_diffusion_pipeline(pipeline_id, task)
            zentorch_pipeline = compile_diffusion_pipeline(pipeline, backend="zentorch")
            zentorch_images = zentorch_pipeline(**inputs, **FAST_DIFFUSION_KWARGS).images

            torch.testing.assert_close(inductor_images, zentorch_images, rtol=1e-3, atol=1e-5)
