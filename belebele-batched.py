import json

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LANGUAGES = ["deu_Latn", "fra_Latn", "spa_Latn", "ita_Latn", "pol_Latn", "por_Latn"]

MODELS = ["google/gemma-3-1b-it"]


def write_pretty_json(file_path, data):
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)
    print(f"wrote {file_path}")


def parse_choice(response):
    if len(response) == 0:
        return None
    elif len(response) == 1:
        return choices.index(response[0]) + 1 if response[0] in choices else None
    elif response[0] in choices and not response[1].isalpha():
        return choices.index(response[0]) + 1
    else:
        return None


prompt_template = {}
prompt_template["deu_Latn"] = """{flores_passage}
Frage: {question}
Antwort A: {mc_answer1}
Antwort B: {mc_answer2}
Antwort C: {mc_answer3}
Antwort D: {mc_answer4}
Richtige Antwort: {correct_answer}"""
prompt_template["eng_Latn"] = """{flores_passage}
Question: {question}
Answer A: {mc_answer1}
Answer B: {mc_answer2}
Answer C: {mc_answer3}
Answer D: {mc_answer4}
Correct answer: {correct_answer}"""
prompt_template["fra_Latn"] = """{flores_passage}
Question: {question}
Réponse A: {mc_answer1}
Réponse B: {mc_answer2}
Réponse C: {mc_answer3}
Réponse D: {mc_answer4}
Réponse correcte: {correct_answer}"""
prompt_template["spa_Latn"] = """{flores_passage}
Pregunta: {question}
Respuesta A: {mc_answer1}
Respuesta B: {mc_answer2}
Respuesta C: {mc_answer3}
Respuesta D: {mc_answer4}
Respuesta correcta: {correct_answer}"""
prompt_template["ita_Latn"] = """{flores_passage}
Domanda: {question}
Risposta A: {mc_answer1}
Risposta B: {mc_answer2}
Risposta C: {mc_answer3}
Risposta D: {mc_answer4}
Risposta corretta: {correct_answer}"""
prompt_template["pol_Latn"] = """{flores_passage}
Pytanie: {question}
Odpowiedź A: {mc_answer1}
Odpowiedź B: {mc_answer2}
Odpowiedź C: {mc_answer3}
Odpowiedź D: {mc_answer4}
Prawidłowa odpowiedź: {correct_answer}"""
prompt_template["por_Latn"] = """{flores_passage}
Pergunta: {question}
Resposta A: {mc_answer1}
Resposta B: {mc_answer2}
Resposta C: {mc_answer3}
Resposta D: {mc_answer4}
Resposta correta: {correct_answer}"""
choices = ["A", "B", "C", "D"]

bs = 6

for language in LANGUAGES:
    dataset_config = {"path": "facebook/belebele", "name": language, "split": "test"}
    dataset = load_dataset(**dataset_config)
    dataset_examples = dataset.select(range(0, 5))
    dataset_prompts = dataset.select(range(5, len(dataset)))

    prompt_examples = "\n\n".join(
        [
            prompt_template[language].format(
                **item, correct_answer=choices[int(item["correct_answer_num"]) - 1]
            )
            for item in dataset_examples
        ]
    )

    for model in MODELS:
        print(f"Evaluating model {model} with language {language}")

        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

        prompts = [
            (
                prompt_examples
                + "\n\n"
                + prompt_template[language].format(**prompt, correct_answer="")
            ).strip()
            for prompt in dataset_prompts
        ]

        result = {
            "dataset": dataset_config,
            "model": model,
            "total": 0,
            "correct": 0,
            "correct_percent": None,
            "prompt_template": prompt_template[language],
            "examples": prompt_examples,
            "questions": [],
        }

        q_correct = q_total = 0
        for start in tqdm(range(0, len(prompts), bs)):
            stop = min(start + bs, len(prompts) - 1)

            prompts_batch = prompts[start:stop]

            encodings = tokenizer(
                prompts_batch, return_tensors="pt", padding="longest", truncation=False
            )
            with torch.no_grad():
                output_ids = model.generate(**encodings)

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
                    print(
                        f"Could not parse {response_raw[len(prompts[sample_no]) + 1 :]}"
                    )

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
                    }
                )
                result["total"] = q_total
                result["correct"] = q_correct
                result["correct_percent"] = q_correct / q_total * 100

        print(
            f"{q_total} questions, {q_correct} correct ({round(q_correct / q_total * 100, 1)}%)"
        )

        write_pretty_json(
            "results/belebe-{}_{}.json".format(model.split("/")[-1], language), result
        )
