import gc
import json
import logging
import os
import shutil
import time

import torch
import torch._dynamo
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

torch._dynamo.config.cache_size_limit = 64

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

LANGUAGES = [
    "eng_Latn",
    "deu_Latn",
    "fra_Latn",
    "spa_Latn",
    "ita_Latn",
    "pol_Latn",
    "por_Latn",
    "est_Latn",
]

MODELS = {
    "mistralai/Mistral-7B-v0.1": {},
    "mistralai/Mistral-7B-Instruct-v0.3": {},
    # "mistralai/Mixtral-8x7B-Instruct-v0.1": {},
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


PROMPT_TEMPLATE = {
    "deu_Latn": """{flores_passage}
Frage: {question}
Antwort A: {mc_answer1}
Antwort B: {mc_answer2}
Antwort C: {mc_answer3}
Antwort D: {mc_answer4}
Richtige Antwort: {correct_answer}""",
    "eng_Latn": """{flores_passage}
Question: {question}
Answer A: {mc_answer1}
Answer B: {mc_answer2}
Answer C: {mc_answer3}
Answer D: {mc_answer4}
Correct answer: {correct_answer}""",
    "fra_Latn": """{flores_passage}
Question: {question}
Réponse A: {mc_answer1}
Réponse B: {mc_answer2}
Réponse C: {mc_answer3}
Réponse D: {mc_answer4}
Réponse correcte: {correct_answer}""",
    "spa_Latn": """{flores_passage}
Pregunta: {question}
Respuesta A: {mc_answer1}
Respuesta B: {mc_answer2}
Respuesta C: {mc_answer3}
Respuesta D: {mc_answer4}
Respuesta correcta: {correct_answer}""",
    "ita_Latn": """{flores_passage}
Domanda: {question}
Risposta A: {mc_answer1}
Risposta B: {mc_answer2}
Risposta C: {mc_answer3}
Risposta D: {mc_answer4}
Risposta corretta: {correct_answer}""",
    "pol_Latn": """{flores_passage}
Pytanie: {question}
Odpowiedź A: {mc_answer1}
Odpowiedź B: {mc_answer2}
Odpowiedź C: {mc_answer3}
Odpowiedź D: {mc_answer4}
Prawidłowa odpowiedź: {correct_answer}""",
    "por_Latn": """{flores_passage}
Pergunta: {question}
Resposta A: {mc_answer1}
Resposta B: {mc_answer2}
Resposta C: {mc_answer3}
Resposta D: {mc_answer4}
Resposta correta: {correct_answer}""",
    "est_Latn": """{flores_passage}
Küsimus: {question}
Vastus A: {mc_answer1}
Vastus B: {mc_answer2}
Vastus C: {mc_answer3}
Vastus D: {mc_answer4}
Õige vastus: {correct_answer}""",
}
CHOICES = ["A", "B", "C", "D"]


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


def write_pretty_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)


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


def results_exist(model_name, language):
    """Check if results file already exists for a model and language combination."""
    output_file_name = "results/belebe-{}_{}.json".format(
        model_name.split("/")[-1], language
    )
    return os.path.exists(output_file_name)


def load_model_and_tokenizer(actual_model_name):
    """Load model and tokenizer for the given model name."""
    logger.info(f"Loading model: {actual_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(actual_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set left padding for decoder-only models like OLMo
    if "olmo" in actual_model_name.lower() or "smollm3" in actual_model_name.lower():
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        actual_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def parse_choice(response):
    if len(response) == 0:
        return None
    elif len(response) == 1:
        return CHOICES.index(response[0]) + 1 if response[0] in CHOICES else None
    elif response[0] in CHOICES and not response[1].isalpha():
        return CHOICES.index(response[0]) + 1
    else:
        return None


current_model_name = ""
model = None
tokenizer = None

for model_name, language_variants in MODELS.items():
    for language in LANGUAGES:
        if language_variants and language in language_variants:
            actual_model_name = language_variants[language]
        elif not language_variants:
            actual_model_name = model_name
        else:
            logger.warning(
                f"Skipping {language} for {model_name} (no variant available)"
            )
            continue

        # Check if results already exist before loading model
        if results_exist(actual_model_name, language):
            logger.info(
                f"Skipping model {model_name} with language {language},"
                " results file exists"
            )
            continue

        # Load model only if needed and different from current
        if current_model_name != actual_model_name:
            if model is not None:
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                clear_huggingface_cache()

            model, tokenizer = load_model_and_tokenizer(actual_model_name)
            current_model_name = actual_model_name

        logger.info(f"Evaluating model {model_name} with language {language}")

        output_file_name = "results/belebe-{}_{}.json".format(
            actual_model_name.split("/")[-1], language
        )

        dataset_config = {
            "path": "facebook/belebele",
            "name": language,
            "split": "test",
        }
        dataset = load_dataset(**dataset_config)
        dataset_examples = dataset.select(range(0, 5))
        dataset_prompts = dataset.select(range(5, len(dataset)))

        prompt_examples = "\n\n".join(
            [
                PROMPT_TEMPLATE[language].format(
                    **item, correct_answer=CHOICES[int(item["correct_answer_num"]) - 1]
                )
                for item in dataset_examples
            ]
        )

        prompts = [
            (
                prompt_examples
                + "\n\n"
                + PROMPT_TEMPLATE[language].format(**prompt, correct_answer="")
            ).strip()
            for prompt in dataset_prompts
        ]

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
                            torch.cuda.get_device_properties(0).total_memory / 1024**3,
                            2,
                        ),
                        "cuda_version": torch.version.cuda,
                    }
                )
            except Exception:
                pass

        batch_size, model_size_b, model_memory_gb = calculate_batch_size(
            model, gpu_info["vram_total_gb"]
        )

        result = {
            "dataset": dataset_config,
            "model": current_model_name,
            "total": 0,
            "correct": 0,
            "correct_percent": None,
            "prompt_template": PROMPT_TEMPLATE[language],
            "examples": prompt_examples,
            "questions": [],
            "gpu_info": gpu_info,
            "batch_size": batch_size,
            "model_size_billions": round(model_size_b, 2),
            "model_memory_gb": round(model_memory_gb, 2),
        }

        q_correct = q_total = 0
        logger.info(
            f"Model: {model_size_b:.2f}B parameters ({model_memory_gb:.2f}GB), "
            f"batch size: {batch_size}"
        )
        for start in tqdm(range(0, len(prompts), batch_size)):
            stop = min(start + batch_size, len(prompts))

            prompts_batch = prompts[start:stop]

            encodings = tokenizer(
                prompts_batch, return_tensors="pt", padding="longest", truncation=False
            ).to(model.device)

            start_time = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **encodings, cache_implementation="offloaded"
                )
            end_time = time.time()

            batch_inference_time = end_time - start_time

            responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for i, response_raw in enumerate(responses):
                sample_no = i + start

                response = response_raw[len(prompts[sample_no]) :]
                response = (
                    response.split("\n")[0].strip()
                    if "\n" in response
                    else response.strip()
                )

                choice = parse_choice(response)

                if choice == int(dataset_prompts[sample_no]["correct_answer_num"]):
                    q_correct += 1
                q_total += 1
                if choice is None:
                    logger.warning(
                        f"Could not parse {response_raw[len(prompts[sample_no]) + 1 :]}"
                    )

                question_inference_time = batch_inference_time / len(responses)

                result["questions"].append(
                    {
                        "question": prompts[sample_no],
                        "answer_raw": response_raw[len(prompts[sample_no]) + 1 :],
                        "answer": choice,
                        "correct_answer": int(
                            dataset_prompts[sample_no]["correct_answer_num"]
                        ),
                        "correct": choice
                        == int(dataset_prompts[sample_no]["correct_answer_num"]),
                        "inference_time_seconds": round(question_inference_time, 3),
                    }
                )

        result["total"] = q_total
        result["correct"] = q_correct
        result["correct_percent"] = q_correct / q_total * 100
        total_inference_time = sum(
            q["inference_time_seconds"] for q in result["questions"]
        )
        result["average_inference_time_seconds"] = round(
            total_inference_time / len(result["questions"]), 3
        )

        logger.info(
            f"{q_total} questions, {q_correct} correct "
            f"({round(q_correct / q_total * 100, 1)}%)"
        )

        write_pretty_json(output_file_name, result)

        if hasattr(model, "past_key_values"):
            model.past_key_values = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
