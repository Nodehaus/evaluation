import argparse
import json
import os
from typing import Any, Dict, List

from deepeval import evaluate
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from multilingual_agent.agent import ModelNotSupported, MultilingualAgent

# LANGUAGES = ["eng"]
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
                    # Check if required keys exist
                    if "name" in tool_call and "arguments" in tool_call:
                        tools.append(
                            ToolCall(
                                name=tool_call["name"],
                                input_parameters=tool_call["arguments"],
                            )
                        )
                    else:
                        print(f"Wrong tool_call format: {tool_call}")

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

        # Save results to JSON file
        self._save_results(result, eval_data, data_path)

        return result

    def _save_results(
        self, result: EvaluationResult, eval_data: Dict[str, Any], data_path: str
    ):
        """Save evaluation results to JSON file."""
        # Extract language code from dataset (default to 'eng')
        language_code = eval_data.get("dataset", {}).get("language", "eng")

        model_name_clean = self.agent.model_name.split("/")[-1]

        # Create results directory with script path as prefix
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Create filename
        filename = f"agent_{model_name_clean}_{language_code}_Latn.json"
        filepath = os.path.join(results_dir, filename)

        # Convert result to JSON-serializable format
        results_dict = {
            "model_name": self.agent.model_name,
            "language": language_code,
            "dataset_path": data_path,
            "evaluation_results": self._serialize_evaluation_result(result),
        }

        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to: {filepath}")

    def _serialize_evaluation_result(self, result: EvaluationResult) -> Dict[str, Any]:
        """Convert EvaluationResult to JSON-serializable dictionary."""
        return {
            "test_results": [
                {
                    "name": test_result.name,
                    "success": test_result.success,
                    "metrics_data": [
                        {
                            "name": metric.name,
                            "threshold": metric.threshold,
                            "success": metric.success,
                            "score": metric.score,
                            "reason": metric.reason,
                        }
                        for metric in test_result.metrics_data
                    ],
                }
                for test_result in result.test_results
            ]
        }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Multilingual Agent Tool Use")
    parser.add_argument(
        "--model", default="HuggingFaceTB/SmolLM3-3B", help="HuggingFace model name"
    )

    args = parser.parse_args()

    # Use default data path with script directory as prefix
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "tools_use_eng.json")

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return

    # Create evaluator
    try:
        evaluator = AgentEvaluator(args.model)
    except ModelNotSupported:
        print("Skipping evaluation for unsupported model.")
        return

    # Run evaluation
    evaluator.run_evaluation(data_path)


if __name__ == "__main__":
    main()
