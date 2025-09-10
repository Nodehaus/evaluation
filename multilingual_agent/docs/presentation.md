---
marp: true
theme: gaia
style: |
    section::before {
      content: url(images/logo.svg);
      position: absolute;
      top: 20px;
      right: 10px;
      width: 320px;
      height: auto;
    }
---

-   Multilingual capabilities of LLMS, our work and others
-   Then anatomy of a simple tool calling agent
-   Chat templates and tool calling
-   DeepEval findings
-   How we score, how our evaluation and metrics works
-   Results
-   Summary and Outlook

---

# About us

-   Nodehaus
-   Julian and Peter

---

# Previous work

-   MAPS: A Multilingual Benchmark for Global Agent Performance and Security
    -   https://arxiv.org/html/2505.15935v1
    -   Dataset not yet published
    -   German, Spanish, Portuguese (Brazil), Japanese, Russian, Italian, Arabic, Hebrew, Korean, and Hindi
-   EuroEval: https://euroeval.com/
    -   13 Monolingual + 4 Multilingual ("European")
    -   No agentic evaluation tasks
-   Nodehaus' The Great European AI Language Championship
    -   https://substack.com/home/post/p-172471752
    -   Testing smaller LLMs on multiple choice question answering
    -   English, German, French, Spanish, Italian, Polish, Portuguese, Estonian

---

# Multilingual LLM performance

Show box plot from our blog post

---

# The mysterious AI Agent

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F58d9f10c985c4eb5d53798dea315f7bb5ab6249e-2401x1000.png&w=3840&q=75

---

# The anatomy of a simple tool calling agent

-   Ask questions about the weather in locations
    -   What's the weather on 12.9.2025 in Munich?
    -   Weather next Monday in Mumbai please
-   Needs two tools: current date + weather API (date + location arguments)
-   Needs to call in correct order with correct arguments
    -   Calculate dates
    -   Place names in English

---

# Code browsing

---

# Chat templates and tool calling

## Qwen3-8B

### System prompt

```
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_current_date", "description": "Get the current date.", "parameters": {"type": "object", "properties": {}}, "return": {"type": "string", "description": "Current date in YYYY-MM-DD format"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
```

### User message + Assistant answer

```
<|im_start|>user
How will the weather be in Munich tomorrow?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_current_date", "arguments": {}}
</tool_call><|im_end|>
```

We need to call the tool here and add messages to the conversation history, one for the tool call, one for the result:

```
  {
    "role": "assistant",
    "tool_calls": [
      {
        "type": "function",
        "function": {
          "name": "get_current_date",
          "arguments": {}
        },
        "id": "I68auS2iR"
      }
    ]
  },
  {
    "role": "tool",
    "content": "2025-09-11",
    "tool_call_id": "I68auS2iR"
  }

```

The tool call result will appear as follows in the prompt:

```
<|im_start|>user
<tool_response>
2025-09-11
</tool_response><|im_end|>
```

## GPT-OSS-20B

### System prompt

```
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-09-10

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.<|end|><|start|>developer<|message|># Tools

## functions

namespace functions {

// Get the current date.
type get_current_date = () => any;

} // namespace functions<|end|>
```

### User message + Assistant answer

```
<|start|>user<|message|>How will the weather be in Munich tomorrow?<|end|>
<|start|>assistant to=functions.get_current_date<|channel|>commentary json<|message|>{}<|call|>
```

We need to call the tool here and add messages to the conversation history, one for the tool call, one for the result:

```
  {
    "role": "assistant",
    "tool_calls": [
      {
        "type": "function",
        "function": {
          "name": "get_current_date",
          "arguments": {}
        },
        "id": "I68auS2iR"
      }
    ]
  },
  {
    "role": "tool",
    "content": "2025-09-11",
    "tool_call_id": "I68auS2iR"
  }

```

The tool call result will appear as follows in the prompt:

```
<|start|>functions.get_current_date to=assistant<|channel|>commentary<|message|>"2025-09-11"<|end|><|start|>assistant
```

---

# Evaluation: First try with DeepEval

-   The open-source LLM evaluation framework: https://deepeval.com/
-   Synthesizer to create evaluation datam supports conversation but not tool calls out-of-the-box
-   Agentic Eval Metrics: Task Completion, Tool Correctness, Argument Correctness
-   LLM as judge for Task Completion and Argument Correctness, needs good judge (default OpenAI GPT-4.1)
-   We decided to implement our own: Claude Code to the rescue!

`You are absolutely right!`

---

# Evaluation Nodehaus: Custom Data + Metrics

-   Claude code workflow:
    -   Prompting to create English data
    -   Human review
    -   Translate language by language
-   Data Categories: No tool calls (10), absolute date (15), relative date (20)
-   Metrics: Tool Correctness and Argument Correctness (direct matches)

---

Browse data + code

---

# Results

-   Show visualizations step-by-step

---

# Summary + Outlook
