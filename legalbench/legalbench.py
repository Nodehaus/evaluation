import gc
import json
import logging
import os
import time

import datasets
import torch
from evaluation import evaluate
from tqdm.auto import tqdm
from utils import generate_prompts

from model_utils import (
    MODELS,
    calculate_batch_size,
    cleanup_model,
    generate_batch_responses,
    generate_single_response,
    get_gpu_info,
    get_model_size_info,
    load_model_and_tokenizer,
)

LANGUAGES = ["eng_Latn"]


def query_model(prompt: str, model, tokenizer, max_new_tokens: int = 20) -> str:
    """
    Query the model with a prompt and return the generated text.
    Uses the same generation approach as belebele-batched.py.

    Args:
        prompt: Input prompt
        model: The loaded model
        tokenizer: The loaded tokenizer
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Generated text response
    """
    return generate_single_response(model, tokenizer, prompt, max_new_tokens)


def extract_answer(response: str) -> str:
    """
    Extract the classification answer from the model response.
    Expected answers: generic, descriptive, suggestive, arbitrary, fanciful

    Args:
        response: The raw model response

    Returns:
        The extracted answer (one of the 5 valid classes)
    """
    response = response.lower().strip()

    # Valid answers
    valid_answers = ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"]

    # Check if any valid answer appears in the response
    for answer in valid_answers:
        if answer in response:
            return answer

    # If no valid answer found, return the first word as fallback
    words = response.split()
    if words:
        return words[0]

    return "unknown"


def write_pretty_json(file_path: str, data: dict):
    """Write data to JSON file with pretty formatting."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)


def results_exist(model_name, language, task_name):
    """Check if results file already exists for a model, language, and task combination."""
    output_file_name = (
        f"results/legalbench-{task_name}-{model_name.split('/')[-1]}_{language}.json"
    )
    return os.path.exists(output_file_name)


def run_evaluation(
    model,
    tokenizer,
    model_name: str,
    language: str = "eng_Latn",
    task_name: str = "abercrombie",
):
    """
    Run evaluation for the abercrombie task using transformers.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        model_name: HuggingFace model name
        language: Language code for evaluation
        task_name: Name of the task to evaluate
    """
    print(f"Running evaluation for task '{task_name}' using model '{model_name}'")

    # Load the dataset
    print("Loading dataset...")
    dataset = datasets.load_dataset(
        "nguha/legalbench", task_name, trust_remote_code=True
    )
    test_df = dataset["test"].to_pandas()

    # Load the prompt template
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_template_path = os.path.join(
        script_dir, f"tasks/{task_name}/base_prompt.txt"
    )
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read()

    # Generate prompts
    print("Generating prompts...")
    prompts = generate_prompts(prompt_template=prompt_template, data_df=test_df)

    # Get GPU and model information
    gpu_info = get_gpu_info()
    model_size_info = get_model_size_info(model)

    # Calculate optimal batch size for batch processing
    batch_size, model_size_b, model_memory_gb = calculate_batch_size(
        model, gpu_info["vram_total_gb"]
    )

    print(
        f"Model: {model_size_info['model_size_billions']}B parameters "
        f"({model_size_info['model_memory_gb']}GB), batch size: {batch_size}"
    )

    # Initialize result structure
    result = {
        "dataset": {"path": "nguha/legalbench", "name": task_name, "split": "test"},
        "model": model_name,
        "total": 0,
        "correct": 0,
        "correct_percent": None,
        "prompt_template": prompt_template,
        "questions": [],
        "gpu_info": gpu_info,
        "model_size_billions": model_size_info["model_size_billions"],
        "model_memory_gb": model_size_info["model_memory_gb"],
        "batch_size": batch_size,
    }

    # Query the model for all prompts using batch processing
    print(f"Querying model for {len(prompts)} samples...")
    responses = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        stop = min(start + batch_size, len(prompts))
        prompts_batch = prompts[start:stop]

        start_time = time.time()
        batch_responses = generate_batch_responses(
            model, tokenizer, prompts_batch, batch_size
        )
        end_time = time.time()

        batch_inference_time = end_time - start_time
        responses.extend(batch_responses)

        # Process each response in the batch
        for i, response in enumerate(batch_responses):
            sample_no = start + i
            question_inference_time = batch_inference_time / len(batch_responses)

            # Extract answer and check correctness
            prediction = extract_answer(response)
            ground_truth = test_df.iloc[sample_no]["answer"]
            is_correct = prediction == ground_truth

            # Add question data to result
            result["questions"].append(
                {
                    "question": prompts[sample_no],
                    "answer_raw": response,
                    "answer": prediction,
                    "correct_answer": ground_truth,
                    "correct": is_correct,
                    "inference_time_seconds": round(question_inference_time, 3),
                }
            )

    # Extract answers from responses
    print("Extracting answers...")
    predictions = [extract_answer(response) for response in responses]

    # Get ground truth answers
    answers = test_df["answer"].tolist()

    # Evaluate performance
    print("Computing evaluation metrics...")
    score = evaluate(task_name, predictions, answers)

    # Update result with final metrics
    result["total"] = len(predictions)
    result["correct"] = sum(1 for pred, ans in zip(predictions, answers) if pred == ans)
    result["correct_percent"] = result["correct"] / result["total"] * 100

    total_inference_time = sum(q["inference_time_seconds"] for q in result["questions"])
    result["average_inference_time_seconds"] = round(
        total_inference_time / len(result["questions"]), 3
    )
    result["balanced_accuracy"] = score

    # Save results to JSON file
    output_file_name = (
        f"results/legalbench-{task_name}-{model_name.split('/')[-1]}_{language}.json"
    )
    write_pretty_json(output_file_name, result)
    print(f"Results saved to {output_file_name}")

    # Print results
    print(f"\nResults for {task_name} using {model_name}:")
    print(f"Balanced Accuracy: {score:.4f}")
    print(
        f"Correct: {result['correct']}/{result['total']} "
        f"({result['correct_percent']:.2f}%)"
    )

    return score, predictions, answers


if __name__ == "__main__":
    # Setup logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Check available devices
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        print(f"Current device: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")

    task_name = "abercrombie"
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
            if results_exist(actual_model_name, language, task_name):
                logger.info(
                    f"Skipping model {model_name} with language {language},"
                    " results file exists"
                )
                continue

            # Load model only if needed and different from current
            if current_model_name != actual_model_name:
                if model is not None:
                    cleanup_model(model, tokenizer)

                model, tokenizer = load_model_and_tokenizer(actual_model_name)
                current_model_name = actual_model_name

            logger.info(f"Evaluating model {model_name} with language {language}")

            # Run the evaluation
            score, predictions, answers = run_evaluation(
                model,
                tokenizer,
                model_name=actual_model_name,
                language=language,
                task_name=task_name,
            )

            logger.info(
                f"Final score for {actual_model_name} ({language}): {score:.4f}"
            )

            # Clear model cache after each evaluation
            if hasattr(model, "past_key_values"):
                model.past_key_values = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
