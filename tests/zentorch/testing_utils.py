import requests
import torch
from diffusers import DiffusionPipeline
from PIL import Image

from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoModelForVision2Seq,
    AutoModelForVisualQuestionAnswering,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.pipelines.audio_utils import ffmpeg_read


SEED = 42
BATCH_SIZE = 2
TEXT_GENERATION_KWARGS = {
    "min_new_tokens": 10,
    "max_new_tokens": 10,
    "return_dict_in_generate": True,
    "output_logits": True,
}
IMAGE_DIFFUSION_KWARGS = {"num_inference_steps": 2, "output_type": "pt"}

SUPPORTED_MODELS_TINY = {
    # text encoder
    "bert": {"hf-internal-testing/tiny-random-bert": ["text-classification"]},
    "xlnet": {"hf-internal-testing/tiny-random-xlnet": ["text-classification"]},
    "roberta": {"hf-internal-testing/tiny-random-roberta": ["text-classification"]},
    "distilbert": {"hf-internal-testing/tiny-random-distilbert": ["text-classification"]},
    # image encoder
    "vit": {"hf-internal-testing/tiny-random-ViTForImageClassification": ["image-classification"]},
}
SUPPORTED_MODELS_TINY_TEXT_GENERATION = {
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
    "blip": {
        "hf-internal-testing/tiny-random-BlipForConditionalGeneration": ["image-to-text", "visual-question-answering"]
    },
    "blip2": {
        "hf-internal-testing/tiny-random-Blip2ForConditionalGeneration": ["image-to-text", "visual-question-answering"]
    },
    # automatic speech recognition
    "whisper": {"openai/whisper-tiny": ["automatic-speech-recognition"]},
}

SUPPORTED_MODELS_TINY_IMAGE_DIFFUSION = {
    # stable diffusion
    "stable-diffusion": {"hf-internal-testing/tiny-stable-diffusion-torch": ["stable-diffusion"]},
    "stable-diffusion-xl": {"hf-internal-testing/tiny-stable-diffusion-xl-pipe": ["stable-diffusion-xl"]},
}


def load_model_or_pipeline(name_or_path: str, task: str):
    torch.manual_seed(SEED)

    if task == "text-classification":
        return AutoModelForSequenceClassification.from_pretrained(name_or_path)
    elif task == "image-classification":
        return AutoModelForImageClassification.from_pretrained(name_or_path)
    elif task == "text-generation":
        return AutoModelForCausalLM.from_pretrained(name_or_path)
    elif task == "text2text-generation":
        return AutoModelForSeq2SeqLM.from_pretrained(name_or_path)
    elif task == "automatic-speech-recognition":
        return AutoModelForSpeechSeq2Seq.from_pretrained(name_or_path)
    elif task == "audio-classification":
        return AutoModelForAudioClassification.from_pretrained(name_or_path)
    elif task == "image-to-text":
        return AutoModelForVision2Seq.from_pretrained(name_or_path)
    elif task == "visual-question-answering":
        return AutoModelForVisualQuestionAnswering.from_pretrained(name_or_path)
    elif task in ["stable-diffusion", "stable-diffusion-xl"]:
        return DiffusionPipeline.from_pretrained(name_or_path)
    else:
        raise ValueError(f"Task {task} not supported")


def get_inputs(model_id: str, task: str):
    if task in ["text-classification", "text-generation", "text2text-generation"]:
        processor = AutoTokenizer.from_pretrained(model_id)
        text = ["This is a test sentence"] * BATCH_SIZE
        dummy_inputs = processor(text=text, return_tensors="pt")

    elif task in ["image-to-text", "image-classification"]:
        processor = AutoImageProcessor.from_pretrained(model_id)
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        images = [Image.open(requests.get(image_url, stream=True).raw)] * BATCH_SIZE
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
        images = [Image.open(requests.get(image_url, stream=True).raw)] * BATCH_SIZE
        dummy_inputs = processor(text=text, images=images, return_tensors="pt")

    elif task in ["stable-diffusion", "stable-diffusion-xl"]:
        dummy_inputs = {"prompt": ["This is a test prompt"] * BATCH_SIZE}

    else:
        raise ValueError(f"Task {task} not supported")

    return dummy_inputs


def load_and_compile_model(model_id, task, backend):
    model = load_model_or_pipeline(model_id, task)

    torch._dynamo.reset()
    model = torch.compile(model, backend=backend)

    return model


def load_and_compile_pipeline(model_id, task, backend):
    pipe = load_model_or_pipeline(model_id, task)

    torch._dynamo.reset()
    pipe.unet = torch.compile(pipe.unet, backend=backend)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, backend=backend)

    return pipe