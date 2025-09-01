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
    clear_huggingface_cache,
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


def extract_answer(response: str, task_name: str) -> str:
    """
    Extract the answer from the model response based on the task type.

    Args:
        response: The raw model response
        task_name: The name of the task to determine extraction method

    Returns:
        The extracted answer
    """
    response = response.strip()

    # Task-specific answer extraction
    if task_name == "abercrombie":
        response_lower = response.lower()
        valid_answers = [
            "generic",
            "descriptive",
            "suggestive",
            "arbitrary",
            "fanciful",
        ]
        for answer in valid_answers:
            if answer in response_lower:
                return answer
        # Fallback to first word
        words = response_lower.split()
        return words[0] if words else "unknown"

    elif (
        task_name
        in [
            "definition_classification",
            "hearsay",
            "textualism_tool_dictionaries",
            "overruling",
        ]
        or task_name.startswith("cuad_")
        or task_name.startswith("maud_")
        or task_name.startswith("opp115_")
    ):
        # Binary classification tasks that expect Yes/No answers
        response_lower = response.lower().strip()
        if "yes" in response_lower:
            return "Yes"
        elif "no" in response_lower:
            return "No"
        # Fallback to first word
        words = response.split()
        return words[0] if words else "No"

    elif task_name == "international_citizenship_questions":
        # This task may have multiple choice answers, extract first word/phrase
        words = response.strip().split()
        return words[0] if words else "unknown"

    elif task_name == "privacy_policy_entailment":
        response_lower = response.lower().strip()
        if "entailment" in response_lower and "not" not in response_lower:
            return "Entailment"
        elif "contradiction" in response_lower:
            return "Contradiction"
        elif "not_entailment" in response_lower or "not entailment" in response_lower:
            return "Not_entailment"
        # Fallback to first word
        words = response.split()
        return words[0] if words else "Not_entailment"

    elif task_name.startswith("ssla_"):
        # SSLA tasks expect names/entities, return the response as-is for complex evaluation
        return response.strip()

    elif task_name == "definition_extraction":
        # Return the response as-is for complex evaluation
        return response.strip()

    else:
        # Generic extraction - return first word or the whole response if short
        words = response.strip().split()
        if len(words) == 1 or len(response.strip()) < 50:
            return response.strip()
        return words[0] if words else "unknown"


def write_pretty_json(file_path: str, data: dict):
    """Write data to JSON file with pretty formatting."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)


def results_exist(model_name, language, task_name):
    """Check if results file already exists for a model, language, and task combination."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_name = os.path.join(
        script_dir,
        f"results/legalbench-{task_name}-{model_name.split('/')[-1]}_{language}.json",
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
    Run evaluation for a specific legalbench task using transformers.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        model_name: HuggingFace model name
        language: Language code for evaluation
        task_name: Name of the task to evaluate
    """
    print(f"Running evaluation for task '{task_name}' using model '{model_name}'")

    # Check if task folder exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    task_dir = os.path.join(script_dir, f"tasks/{task_name}")
    if not os.path.exists(task_dir):
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

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
            prediction = extract_answer(response, task_name)
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
    predictions = [extract_answer(response, task_name) for response in responses]

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_name = os.path.join(
        script_dir,
        f"results/legalbench-{task_name}-{model_name.split('/')[-1]}_{language}.json",
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


def get_global_automatic_tasks():
    """
    Get all tasks that have automatic_evaluation=true and jurisdiction=global.

    Returns:
        List of task names
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    task_analysis_path = os.path.join(script_dir, "legalbench_task_analysis.json")

    with open(task_analysis_path, "r") as f:
        tasks = json.load(f)

    global_tasks = [
        task["task"]
        for task in tasks
        if task["automatic_evaluation"] == "true" and task["jurisdiction"] == "global"
    ]

    return global_tasks


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

    # Get all global automatic tasks
    tasks_to_evaluate = get_global_automatic_tasks()
    print(f"Found {len(tasks_to_evaluate)} global automatic evaluation tasks")

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

            for task_name in tasks_to_evaluate:
                # Check if results already exist before loading model
                if results_exist(actual_model_name, language, task_name):
                    logger.info(
                        f"Skipping task {task_name} for model {model_name} with language {language},"
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

                logger.info(
                    f"Evaluating model {model_name} with language {language} on task {task_name}"
                )

                # Run the evaluation
                score, predictions, answers = run_evaluation(
                    model,
                    tokenizer,
                    model_name=actual_model_name,
                    language=language,
                    task_name=task_name,
                )

                logger.info(f"Task {task_name} - Score: {score:.4f}")

                # Clear model cache after each evaluation
                if hasattr(model, "past_key_values"):
                    model.past_key_values = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
