import gc
import logging
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

BASE_MODELS = {"pbouda/finetune-cpt-test": "unsloth/mistral-7b-v0.3"}

MODELS = {
    "mistralai/Mistral-7B-v0.3": {},
    "pbouda/finetune-cpt-test": {},
    "mistralai/Mistral-7B-Instruct-v0.3": {},
    "utter-project/EuroLLM-9B": {},
    "utter-project/EuroLLM-1.7B": {},
    "google/gemma-3-1b-pt": {},
    "google/gemma-3-1b-it": {},
    "google/gemma-3-4b-pt": {},
    "google/gemma-3-4b-it": {},
    "google/gemma-3-12b-pt": {},
    "google/gemma-3-12b-it": {},
    "hplt-monolingual": {
        "deu_Latn": "HPLT/hplt2c_deu_checkpoints",
        "fra_Latn": "HPLT/hplt2c_fra_checkpoints",
        "spa_Latn": "HPLT/hplt2c_spa_checkpoints",
        "ita_Latn": "HPLT/hplt2c_ita_checkpoints",
        "pol_Latn": "HPLT/hplt2c_pol_checkpoints",
        "por_Latn": "HPLT/hplt2c_por_checkpoints",
        "eng_Latn": "HPLT/hplt2c_eng_checkpoints",
        "est_Latn": "HPLT/hplt2c_est_checkpoints",
    },
    "allenai/OLMo-2-1124-13B-Instruct": {},
    "allenai/OLMo-2-1124-13B": {},
    "allenai/OLMo-2-1124-7B-Instruct": {},
    "allenai/OLMo-2-1124-7B": {},
    "HuggingFaceTB/SmolLM3-3B-Base": {},
    "HuggingFaceTB/SmolLM3-3B": {},
}


def load_model_and_tokenizer(model_name: str):
    """
    Load model and tokenizer for the given model name.
    Uses the same approach as belebele-batched.py for consistent generation behavior.
    Supports LoRA models via PEFT library.

    Args:
        model_name: HuggingFace model name

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")

    # Determine the model name to use for tokenizer and base model
    is_lora = model_name in BASE_MODELS
    base_model_name = BASE_MODELS[model_name] if is_lora else model_name

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set left padding for decoder-only models like OLMo
    if "olmo" in base_model_name.lower() or "smollm3" in base_model_name.lower():
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load LoRA adapter if this is a LoRA model
    if is_lora:
        model = PeftModel.from_pretrained(model, model_name)

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def calculate_batch_size(model, vram_gb):
    """Calculate optimal batch size based on model size and available VRAM."""
    # Get actual model size and memory usage
    total_params = sum(p.numel() for p in model.parameters())
    model_size_b = total_params / 1e9

    # Get bytes per parameter from dtype
    sample_param = next(model.parameters())
    bytes_per_param = sample_param.element_size()
    model_memory_gb = total_params * bytes_per_param / 1e9

    if vram_gb is None or vram_gb < 4:
        return 1, model_size_b, model_memory_gb

    # Calculate batch size based on available memory after model loading
    # Reserve some VRAM for model overhead, activations, and safety margin
    available_memory_gb = vram_gb - model_memory_gb - 1  # 1GB safety margin

    if available_memory_gb <= 0:
        batch_size = 1
    else:
        # Estimate memory per batch item (rough approximation for inference)
        # Each batch item uses roughly 0.7GB for a typical prompt length
        memory_per_batch_item = 0.8
        batch_size = max(1, int(available_memory_gb / memory_per_batch_item))

    return batch_size, model_size_b, model_memory_gb


def generate_batch_responses(model, tokenizer, prompts, batch_size=1):
    """
    Generate responses for a batch of prompts using the same approach as belebele-batched.py.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompts: List of prompt strings
        batch_size: Batch size for generation

    Returns:
        List of response strings (without the original prompt)
    """
    all_responses = []

    for start in range(0, len(prompts), batch_size):
        stop = min(start + batch_size, len(prompts))
        prompts_batch = prompts[start:stop]

        encodings = tokenizer(
            prompts_batch, return_tensors="pt", padding="longest", truncation=False
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**encodings, cache_implementation="offloaded")

        responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # Extract only the generated part (remove original prompt)
        for i, response_raw in enumerate(responses):
            response = response_raw[len(prompts[start + i]) :]
            # Clean up response (take first line if multiple lines)
            response = (
                response.split("\n")[0].strip()
                if "\n" in response
                else response.strip()
            )
            all_responses.append(response)

    return all_responses


def generate_single_response(model, tokenizer, prompt, max_new_tokens=20):
    """
    Generate a single response using the batched approach with batch_size=1.
    This ensures consistent behavior with the batched generation.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: Single prompt string
        max_new_tokens: Maximum number of tokens to generate (for compatibility)

    Returns:
        Generated response string (without the original prompt)
    """
    return generate_batch_responses(model, tokenizer, [prompt], batch_size=1)[0]


def cleanup_model(model, tokenizer):
    """Clean up model and tokenizer to free memory."""
    if model is not None:
        if hasattr(model, "past_key_values"):
            model.past_key_values = None
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def clear_huggingface_cache():
    """Clear HuggingFace model cache to free up disk space."""
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        try:
            # Clear transformers cache
            transformers_cache = os.path.join(cache_dir, "transformers")
            if os.path.exists(transformers_cache):
                shutil.rmtree(transformers_cache)
                logger.info("Cleared HuggingFace transformers cache")

            # Clear hub cache (downloaded models)
            hub_cache = os.path.join(cache_dir, "hub")
            if os.path.exists(hub_cache):
                shutil.rmtree(hub_cache)
                logger.info("Cleared HuggingFace hub cache")

        except Exception as e:
            logger.warning(f"Failed to clear HuggingFace cache: {e}")
    else:
        logger.info("HuggingFace cache directory not found")


def get_gpu_info():
    """Get GPU information."""
    gpu_info = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_model": None,
        "vram_total_gb": None,
        "cuda_version": None,
    }

    if torch.cuda.is_available():
        try:
            gpu_info.update(
                {
                    "gpu_model": torch.cuda.get_device_name(0),
                    "vram_total_gb": round(
                        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
                    ),
                    "cuda_version": torch.version.cuda,
                }
            )
        except Exception:
            pass

    return gpu_info


def get_model_size_info(model):
    """Get model size information."""
    total_params = sum(p.numel() for p in model.parameters())
    model_size_b = total_params / 1e9
    sample_param = next(model.parameters())
    bytes_per_param = sample_param.element_size()
    model_memory_gb = total_params * bytes_per_param / 1e9

    return {
        "model_size_billions": round(model_size_b, 2),
        "model_memory_gb": round(model_memory_gb, 2),
        "total_params": total_params,
    }
