import os

import PIL
import requests
import torch
import zentorch  # noqa: F401
from diffusers import AutoPipelineForText2Image

from optimum.exporters.tasks import TasksManager
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.pipelines.audio_utils import ffmpeg_read


# to avoid fast tokenizers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"

SEED = 42
BATCH_SIZE = 2

FAST_TEXT_GENERATION_KWARGS = {
    "min_new_tokens": 2,
    "max_new_tokens": 2,
    "output_logits": True,
    "return_dict_in_generate": True,
}
FAST_DIFFUSION_KWARGS = {"num_inference_steps": 2, "output_type": "pt"}

SUPPORTED_COMMON_MODELS_TINY = {
    # text encoder
    "bert": {"hf-internal-testing/tiny-random-bert": ["text-classification"]},
    "xlnet": {"hf-internal-testing/tiny-random-xlnet": ["text-classification"]},
    "roberta": {"hf-internal-testing/tiny-random-roberta": ["text-classification"]},
    "distilbert": {"hf-internal-testing/tiny-random-distilbert": ["text-classification"]},
    # image encoder
    "vit": {"hf-internal-testing/tiny-random-ViTForImageClassification": ["image-classification"]},
}
SUPPORTED_TEXT_GENERATION_MODELS_TINY = {
    # text decoder
    "gemma": {"fxmarty/tiny-random-GemmaForCausalLM": ["text-generation"]},
    "phi": {"hf-internal-testing/tiny-random-PhiForCausalLM": ["text-generation"]},
    "opt": {"hf-internal-testing/tiny-random-OPTForCausalLM": ["text-generation"]},
    "mpt": {"hf-internal-testing/tiny-random-MptForCausalLM": ["text-generation"]},
    "gpt2": {"hf-internal-testing/tiny-random-GPT2LMHeadModel": ["text-generation"]},
    "gptj": {"hf-internal-testing/tiny-random-GPTJForCausalLM": ["text-generation"]},
    "bloom": {"hf-internal-testing/tiny-random-BloomForCausalLM": ["text-generation"]},
    "llama": {"hf-internal-testing/tiny-random-LlamaForCausalLM": ["text-generation"]},
    "falcon": {"hf-internal-testing/tiny-random-FalconForCausalLM": ["text-generation"]},
    "gpt-neo": {"hf-internal-testing/tiny-random-GPTNeoForCausalLM": ["text-generation"]},
    "mistral": {"hf-internal-testing/tiny-random-MistralForCausalLM": ["text-generation"]},
    "gpt-neox": {"hf-internal-testing/tiny-random-GPTNeoXForCausalLM": ["text-generation"]},
    "starcoder2": {"hf-internal-testing/tiny-random-Starcoder2ForCausalLM": ["text-generation"]},
    "gpt-bigcode": {"hf-internal-testing/tiny-random-GPTBigCodeForCausalLM": ["text-generation"]},
    # text encoder-decoder
    "t5": {"hf-internal-testing/tiny-random-t5": ["text2text-generation"]},
    "bart": {"hf-internal-testing/tiny-random-bart": ["text2text-generation"]},
    # image encoder text decoder
    "llava": {"IlyasMoutawwakil/tiny-random-LlavaForConditionalGeneration": ["image-to-text"]},
    "blip": {"hf-internal-testing/tiny-random-BlipForConditionalGeneration": ["image-to-text"]},
    "blip2": {"hf-internal-testing/tiny-random-Blip2ForConditionalGeneration": ["image-to-text"]},
    # automatic speech recognition
    "whisper": {"openai/whisper-tiny": ["automatic-speech-recognition"]},
}

SUPPORTED_DIFFUSION_PIPELINES_TINY = {
    "stable-diffusion": {"hf-internal-testing/tiny-stable-diffusion-torch": ["text-to-image"]},
    "stable-diffusion-xl": {"hf-internal-testing/tiny-stable-diffusion-xl-pipe": ["text-to-image"]},
}


def load_transformers_model(model_id: str, task: str):
    torch.manual_seed(SEED)
    model_class = TasksManager().get_model_class_for_task(task, library="transformers", framework="pt")
    return model_class.from_pretrained(model_id)


def load_diffusion_pipeline(pipeline_id: str, task: str):
    torch.manual_seed(SEED)
    if task == "text-to-image":
        return AutoPipelineForText2Image.from_pretrained(pipeline_id)
    else:
        raise ValueError(f"Task {task} not supported")


def get_transformers_model_inputs(model_id: str, task: str):
    if task in ["text-classification", "text-generation", "text2text-generation"]:
        processor = AutoTokenizer.from_pretrained(model_id)
        text = ["This is a test sentence"] * BATCH_SIZE
        dummy_inputs = processor(text=text, return_tensors="pt")

    elif task in ["image-to-text", "image-classification"]:
        processor = AutoImageProcessor.from_pretrained(model_id)
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        images = [PIL.Image.open(requests.get(image_url, stream=True).raw)] * BATCH_SIZE
        dummy_inputs = processor(images=images, return_tensors="pt")

    elif task in ["audio-classification", "automatic-speech-recognition"]:
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
        audios = [ffmpeg_read(requests.get(audio_url).content, processor.sampling_rate)] * BATCH_SIZE
        dummy_inputs = processor(raw_speech=audios, return_tensors="pt")

    elif task == "visual-question-answering":
        processor = AutoProcessor.from_pretrained(model_id)
        text = ["What is in the image?"] * BATCH_SIZE
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        images = [PIL.Image.open(requests.get(image_url, stream=True).raw)] * BATCH_SIZE
        dummy_inputs = processor(text=text, images=images, return_tensors="pt")

    else:
        raise ValueError(f"Task {task} not supported")

    return dummy_inputs


def get_diffusion_pipeline_inputs(pipeline_id: str, task: str):
    dummy_inputs = {}

    if task == "text-to-image":
        dummy_inputs["prompt"] = ["This is a test prompt"] * BATCH_SIZE
    else:
        raise ValueError(f"Task {task} not supported")

    return dummy_inputs


def compile_transformers_model(model, backend):
    torch.manual_seed(SEED)
    torch._dynamo.reset()

    model = torch.compile(model, backend=backend)
    model.eval()

    return model


def compile_diffusion_pipeline(pipeline, backend):
    torch.manual_seed(SEED)
    torch._dynamo.reset()

    pipeline.unet = torch.compile(pipeline.unet, backend=backend)
    pipeline.vae.decoder = torch.compile(pipeline.vae.decoder, backend=backend)

    return pipeline
