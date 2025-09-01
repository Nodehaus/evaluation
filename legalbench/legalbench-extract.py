#!/usr/bin/env python3

import json
import os
import sys

# Add the legalbench directory to the path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'legalbench'))

import datasets
from utils import generate_prompts


def extract_legalbench_samples(task_name="abercrombie", num_samples=100):
    """
    Extract first N samples from legalbench evaluation and save to JSON.
    
    Args:
        task_name: Name of the legalbench task to extract from
        num_samples: Number of samples to extract (default: 100)
    
    Returns:
        List of dicts with "prompt" and "correct_answer" fields
    """
    print(f"Loading legalbench dataset for task '{task_name}'...")
    
    # Load the dataset
    dataset = datasets.load_dataset(
        "nguha/legalbench", task_name, trust_remote_code=True
    )
    test_df = dataset["test"].to_pandas()
    
    # Load the prompt template
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_template_path = os.path.join(
        script_dir, f"legalbench/tasks/{task_name}/base_prompt.txt"
    )
    
    if not os.path.exists(prompt_template_path):
        raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")
    
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read()
    
    print(f"Generating prompts for first {num_samples} samples...")
    
    # Limit to first num_samples
    limited_df = test_df.head(num_samples)
    
    # Generate prompts
    prompts = generate_prompts(prompt_template=prompt_template, data_df=limited_df)
    
    # Extract samples
    samples = []
    for i in range(len(prompts)):
        sample = {
            "prompt": prompts[i],
            "correct_answer": limited_df.iloc[i]["answer"]
        }
        samples.append(sample)
    
    return samples


def main():
    """Extract samples and save to JSON file."""
    task_name = "abercrombie"
    num_samples = 100
    
    print(f"Extracting {num_samples} samples from legalbench task '{task_name}'...")
    
    # Extract samples
    samples = extract_legalbench_samples(task_name, num_samples)
    
    # Save to JSON file
    output_file = f"legalbench-{task_name}-{num_samples}samples.json"
    
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"Saved {len(samples)} samples to {output_file}")
    print(f"Sample structure: {list(samples[0].keys())}")
    print(f"First sample preview:")
    print(f"  Prompt: {samples[0]['prompt'][:100]}...")
    print(f"  Correct answer: {samples[0]['correct_answer']}")


if __name__ == "__main__":
    main()