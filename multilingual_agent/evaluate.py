import argparse
import json
import os
from typing import Any, Dict, List

from deepeval import evaluate
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from multilingual_agent.agent import ModelNotSupported, MultilingualAgent

# MODELS = {
#     "HuggingFaceTB/SmolLM3-3B": {},
#     "Qwen/Qwen3-4B": {},
#     "Qwen/Qwen3-8B": {},
#     "Qwen/Qwen3-14B": {},
#     "mistralai/Mistral-Nemo-Instruct-2407": {},
#     "openai/gpt-oss-20b": {},
# }


class AgentEvaluator:
    """Evaluator for the multilingual agent using DeepEval metrics."""

    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM3-3B"):
        self.agent = MultilingualAgent(model_name=model_name)
        self.tool_correctness_metric = ToolCorrectnessMetric(threshold=0.7)

    def load_evaluation_data(self, data_path: str) -> Dict[str, Any]:
        """Load evaluation dataset from JSON file."""
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def extract_tools_from_conversation(
        self, conversation: List[Dict[str, Any]]
    ) -> List[ToolCall]:
        """Extract expected tool calls from reference conversation."""
        tools = []

        for turn in conversation:
            if turn["role"] == "assistant" and "tool_calls" in turn:
                for tool_call in turn["tool_calls"]:
                    tools.append(
                        ToolCall(
                            name=tool_call["name"],
                            input_parameters=tool_call["parameters"],
                        )
                    )

        return tools

    def extract_actual_tools_from_conversation(
        self, conversation: List[Dict[str, Any]]
    ) -> List[ToolCall]:
        """Extract actual tool calls made by the agent."""
        tools = []

        for turn in conversation:
            if turn["role"] == "assistant" and "tool_calls" in turn:
                for tool_call in turn["tool_calls"]:
                    tools.append(
                        ToolCall(
                            name=tool_call["name"],
                            input_parameters=tool_call["parameters"],
                        )
                    )

        return tools

    def evaluate_conversation(self, eval_item: Dict[str, Any]) -> LLMTestCase:
        """Evaluate a single conversation item."""
        conversation = eval_item["conversation"]
        requires_tool_use = eval_item["requires_tool_use"]

        # Get the initial user message
        user_message = conversation[0]["content"]

        # Generate agent response
        agent_conversation = self.agent.chat(user_message)

        # Extract expected and actual tool calls
        if requires_tool_use:
            expected_tools = self.extract_tools_from_conversation(conversation)
        else:
            # For non-tool conversations, expected tools is empty list
            expected_tools = []

        actual_tools = self.extract_actual_tools_from_conversation(agent_conversation)

        # Get the final assistant response
        final_response = ""
        for turn in reversed(agent_conversation):
            if turn["role"] == "assistant":
                final_response = turn["content"]
                break

        # Create test case for DeepEval
        test_case = LLMTestCase(
            input=user_message,
            actual_output=final_response,
            expected_output="",  # Not used for tool correctness
            tools_called=actual_tools,
            expected_tools=expected_tools,
        )

        return test_case

    def run_evaluation(self, data_path: str) -> EvaluationResult:
        """Run full evaluation on the dataset."""
        print(f"Loading evaluation data from {data_path}")
        eval_data = self.load_evaluation_data(data_path)

        test_cases = []
        skipped = 0

        conversations_count = len(eval_data["conversations"])
        print(f"Processing {conversations_count} conversations...")

        for i, eval_item in enumerate(eval_data["conversations"]):
            progress = f"{i + 1}/{conversations_count}"
            print(f"Processing conversation {eval_item['id']} ({progress})")

            test_case = self.evaluate_conversation(eval_item)
            test_cases.append(test_case)
            tool_status = "with tools" if eval_item["requires_tool_use"] else "no tools"
            print(f"  Processed ({tool_status})")

        print(
            f"\nEvaluating {len(test_cases)} test cases with Tool Correctness metric..."
        )
        print(f"Skipped {skipped} conversations")

        # Run evaluation
        result = evaluate(test_cases, [self.tool_correctness_metric])
        return result


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Multilingual Agent Tool Use")
    parser.add_argument(
        "--data",
        default="data/tools_use_eng.json",
        help="Path to evaluation data JSON file",
    )

    parser.add_argument(
        "--model", default="HuggingFaceTB/SmolLM3-3B", help="HuggingFace model name"
    )

    args = parser.parse_args()

    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        return

    # Create evaluator
    try:
        evaluator = AgentEvaluator(args.model)
    except ModelNotSupported:
        print("Skipping evaluation for unsupported model.")
        return

    # Run evaluation
    evaluator.run_evaluation(args.data)


if __name__ == "__main__":
    main()
