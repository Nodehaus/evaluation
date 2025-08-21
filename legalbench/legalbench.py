import json
import os
import time

import datasets
import torch
from evaluation import evaluate
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate_prompts


def load_model_and_tokenizer(model_name: str):
    """
    Load the transformers model and tokenizer.

    Args:
        model_name: HuggingFace model name

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model and tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def query_model(prompt: str, model, tokenizer, max_new_tokens: int = 20) -> str:
    """
    Query the model with a prompt and return the generated text.

    Args:
        prompt: Input prompt
        model: The loaded model
        tokenizer: The loaded tokenizer
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Generated text response
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Move to same device as model
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for evaluation
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode the response (only the new tokens)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return response


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


def run_evaluation(model_name: str, task_name: str = "abercrombie"):
    """
    Run evaluation for the abercrombie task using transformers.

    Args:
        model_name: HuggingFace model name
        task_name: Name of the task to evaluate
    """
    print(f"Running evaluation for task '{task_name}' using model '{model_name}'")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load the dataset
    print("Loading dataset...")
    dataset = datasets.load_dataset(
        "nguha/legalbench", task_name, trust_remote_code=True
    )
    test_df = dataset["test"].to_pandas()

    # Load the prompt template
    prompt_template_path = f"tasks/{task_name}/base_prompt.txt"
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read()

    # Generate prompts
    print("Generating prompts...")
    prompts = generate_prompts(prompt_template=prompt_template, data_df=test_df)

    # Get GPU information
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

    # Get model size information
    total_params = sum(p.numel() for p in model.parameters())
    model_size_b = total_params / 1e9
    sample_param = next(model.parameters())
    bytes_per_param = sample_param.element_size()
    model_memory_gb = total_params * bytes_per_param / 1e9

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
        "model_size_billions": round(model_size_b, 2),
        "model_memory_gb": round(model_memory_gb, 2),
    }

    # Query the model for all prompts
    print(f"Querying model for {len(prompts)} samples...")
    responses = []
    for i, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        start_time = time.time()
        response = query_model(prompt, model, tokenizer)
        end_time = time.time()

        responses.append(response)
        inference_time = end_time - start_time

        # Extract answer and check correctness
        prediction = extract_answer(response)
        ground_truth = test_df.iloc[i]["answer"]
        is_correct = prediction == ground_truth

        # Add question data to result
        result["questions"].append(
            {
                "question": prompt,
                "answer_raw": response,
                "answer": prediction,
                "correct_answer": ground_truth,
                "correct": is_correct,
                "inference_time_seconds": round(inference_time, 3),
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
        f"results/legalbench-{task_name}-{model_name.split('/')[-1]}_eng_Latn.json"
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
    # Check available devices
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        print(f"Current device: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")

    model_name = "google/gemma-3-1b-it"

    print(f"Using model: {model_name}")

    # Run the evaluation
    score, predictions, answers = run_evaluation(model_name=model_name)

    print(f"Final score: {score:.4f}")
