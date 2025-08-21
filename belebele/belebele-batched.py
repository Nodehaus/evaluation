import gc
import json
import logging
import os
import time

import torch
import torch._dynamo
from datasets import load_dataset
from tqdm import tqdm

from model_utils import (
    MODELS,
    calculate_batch_size,
    cleanup_model,
    clear_huggingface_cache,
    generate_batch_responses,
    get_gpu_info,
    get_model_size_info,
    load_model_and_tokenizer,
)

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


def write_pretty_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)



def results_exist(model_name, language):
    """Check if results file already exists for a model and language combination."""
    output_file_name = "results/belebe-{}_{}.json".format(
        model_name.split("/")[-1], language
    )
    return os.path.exists(output_file_name)


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
                cleanup_model(model, tokenizer)
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

        gpu_info = get_gpu_info()
        model_size_info = get_model_size_info(model)

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
            "model_size_billions": model_size_info["model_size_billions"],
            "model_memory_gb": model_size_info["model_memory_gb"],
        }

        q_correct = q_total = 0
        logger.info(
            f"Model: {model_size_info['model_size_billions']}B parameters "
            f"({model_size_info['model_memory_gb']}GB), batch size: {batch_size}"
        )
        for start in tqdm(range(0, len(prompts), batch_size)):
            stop = min(start + batch_size, len(prompts))
            prompts_batch = prompts[start:stop]

            start_time = time.time()
            batch_responses = generate_batch_responses(
                model, tokenizer, prompts_batch, batch_size
            )
            end_time = time.time()

            batch_inference_time = end_time - start_time

            for i, response in enumerate(batch_responses):
                sample_no = start + i

                choice = parse_choice(response)

                if choice == int(dataset_prompts[sample_no]["correct_answer_num"]):
                    q_correct += 1
                q_total += 1
                if choice is None:
                    logger.warning(f"Could not parse {response}")

                question_inference_time = batch_inference_time / len(batch_responses)

                result["questions"].append(
                    {
                        "question": prompts[sample_no],
                        "answer_raw": response,
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

        # Clear model cache after each language
        if hasattr(model, "past_key_values"):
            model.past_key_values = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
