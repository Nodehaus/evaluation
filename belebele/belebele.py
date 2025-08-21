import copy

import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm

LANGUAGES = ["deu_Latn", "fra_Latn", "spa_Latn", "ita_Latn", "pol_Latn", "por_Latn"]

MODELS = ["google/gemma-3-1b-it"]

# Do we need this?
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.1,
    "repetition_penalty": 1.18,
    "top_k": 40,
    "do_sample": True,
    "max_new_tokens": 5,
    # "pad_token_id": pipeline.tokenizer.eos_token_id,
}

PROMPT_TEMPLATE = """{flores_passage}
Question: {question}
Answer A: {mc_answer1}
Answer B: {mc_answer2}
Answer C: {mc_answer3}
Answer D: {mc_answer4}
Correct answer: {correct_answer}"""

CHOICES = ["A", "B", "C", "D"]


def parse_choice(response):
    if len(response) == 1:
        return CHOICES.index(response[0]) + 1 if response[0] in CHOICES else None
    elif response[0] in CHOICES and not response[1].isalpha():
        return CHOICES.index(response[0]) + 1
    else:
        return None


for language in LANGUAGES:
    dataset = load_dataset(path="facebook/belebele", name=language, split="test")

    # Select a subset of the dataset for prompting
    dataset_examples = dataset.select(range(0, 5))
    dataset_prompts = dataset.select(range(5, len(dataset)))
    prompt_examples = "\n\n".join(
        [
            PROMPT_TEMPLATE.format(
                **item, correct_answer=CHOICES[int(item["correct_answer_num"]) - 1]
            )
            for item in dataset_examples
        ]
    )
    for model in MODELS:
        print(f"Evaluating model {model} with language {language}")
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Format the first five rows as examples for 5-shot prompting
        CHOICES = ["A", "B", "C", "D"]

        # Loop through prompts and evaluate model responses
        q_correct = q_total = 0
        for rowNo, row in enumerate(tqdm(dataset_prompts)):
            prompt = (
                prompt_examples
                + "\n\n"
                + PROMPT_TEMPLATE.format(**row, correct_answer="")
            ).strip()
            generation_config = copy.copy(GENERATION_CONFIG)
            generation_config["pad_token_id"] = pipeline.tokenizer.eos_token_id
            response = pipeline(prompt)[0]["generated_text"][len(prompt) :]
            # response = pipeline(prompt, **GENERATION_CONFIG)[0]["generated_text"][
            #     len(prompt) :
            # ]

            if "\n" in response:
                response = response.split("\n")[0]

            choice = parse_choice(response.strip())
            # print(choice)
            # print(row["correct_answer_num"])
            # print("---")
            if choice == int(row["correct_answer_num"]):
                q_correct += 1
            q_total += 1

        print(
            f"{q_total} questions, {q_correct} correct ({round(q_correct / q_total * 100, 1)}%)"
        )
