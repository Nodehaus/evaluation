# Nodehaus Multilingual LLM Evaluation

Some evaluations on multilingual tasks for LLMs. For now we evaluate with [Belebele](https://github.com/facebookresearch/belebele).

## Install uv and dependencies

```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
$ uv sync
```

## Run script

We have a simplifed Belebele evaluation runner to test individual models and learn about the behaviour:

```
$ uv run belebele.py
```

Or batched to make it faster, this will run all models and languages that are listed in the code and write the results to JSON files (if they don't exist):

```
$ uv run belebele-batched.py
```

## Run analysis

```
$ uv run streamlit run analysis.py
```
