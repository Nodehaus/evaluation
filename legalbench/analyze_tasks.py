#!/usr/bin/env python3
import json
import os
import re


def analyze_task(task_dir):
    """Analyze a single task for automatic evaluation and jurisdiction"""
    task_name = os.path.basename(task_dir)

    # Initialize result
    result = {
        "task": task_name,
        "automatic_evaluation": "false",
        "jurisdiction": "unsure",
    }

    # Try to read README (try both .md and .MD)
    readme_content = ""
    for readme_name in ["README.md", "README.MD"]:
        readme_path = os.path.join(task_dir, readme_name)
        if os.path.exists(readme_path):
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    readme_content = f.read()
                break
            except Exception as e:
                print(f"Error reading {readme_path}: {e}")

    # Try to read base_prompt.txt
    base_prompt_content = ""
    base_prompt_path = os.path.join(task_dir, "base_prompt.txt")
    if os.path.exists(base_prompt_path):
        try:
            with open(base_prompt_path, "r", encoding="utf-8") as f:
                base_prompt_content = f.read()
        except Exception as e:
            print(f"Error reading {base_prompt_path}: {e}")

    combined_content = readme_content + "\n" + base_prompt_content
    combined_lower = combined_content.lower()

    # Determine automatic evaluation
    # Look for task type indicators in README
    automatic_task_types = [
        "binary classification",
        "3-way classification",
        "4-way classification",
        "5-way classification",
        "6-way classification",
        "classification",
        "yes/no",
        "multiple choice",
        "extraction",  # Usually extracting specific factual info
    ]

    manual_task_types = ["open generation", "open-ended", "subjective"]

    # Check task type field specifically
    task_type_match = re.search(r"\*\*task type\*\*:\s*([^\n]+)", combined_lower)
    if task_type_match:
        task_type = task_type_match.group(1).strip()
        for auto_type in automatic_task_types:
            if auto_type in task_type:
                result["automatic_evaluation"] = "true"
                break
        else:
            for manual_type in manual_task_types:
                if manual_type in task_type:
                    result["automatic_evaluation"] = "false"
                    break

    # Determine jurisdiction
    # Look for specific country/jurisdiction indicators
    us_indicators = [
        "united states",
        "u.s.",
        "us ",
        "federal circuit",
        "supreme court",
        "circuit court",
        "federal court",
        "american",
        "usa",
        "federal",
        "delaware",
        "california",
        "new york",
        "texas",
        "state court",
    ]

    canada_indicators = [
        "canada",
        "canadian",
        "tax court of canada",
        "ontario",
        "quebec",
        "british columbia",
        "alberta",
        "manitoba",
        "saskatchewan",
    ]

    international_indicators = [
        "international",
        "global",
        "world",
        "across the world",
        "multiple countries",
        "globalcit",
        "worldwide",
    ]

    # Check for jurisdiction indicators
    # Special cases first
    if task_name == "abercrombie":
        result["jurisdiction"] = "US"  # US trademark law case
    elif task_name.startswith("learned_hands_"):
        result["jurisdiction"] = "US"  # Primarily US legal aid platform
    elif any(indicator in combined_lower for indicator in international_indicators):
        result["jurisdiction"] = "global"
    elif any(indicator in combined_lower for indicator in canada_indicators):
        result["jurisdiction"] = "Canada"
    elif any(indicator in combined_lower for indicator in us_indicators):
        result["jurisdiction"] = "US"

    return result


def main():
    tasks_dir = "/home/pbouda/Projects/poio-eval/legalbench/tasks"
    results = []

    # Get all task directories
    for task_name in sorted(os.listdir(tasks_dir)):
        task_path = os.path.join(tasks_dir, task_name)
        if os.path.isdir(task_path):
            print(f"Analyzing {task_name}...")
            result = analyze_task(task_path)
            results.append(result)

    # Print results as Python list
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS:")
    print("=" * 50)
    print("[")
    for i, result in enumerate(results):
        comma = "," if i < len(results) - 1 else ""
        print(
            f'  {{"task": "{result["task"]}", "automatic_evaluation": "{result["automatic_evaluation"]}", "jurisdiction": "{result["jurisdiction"]}"}}{comma}'
        )
    print("]")

    # Also save to file
    with open("/home/pbouda/Projects/poio-eval/task_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAnalyzed {len(results)} tasks")
    print(f"Results also saved to: /home/pbouda/Projects/poio-eval/task_analysis.json")


if __name__ == "__main__":
    main()
