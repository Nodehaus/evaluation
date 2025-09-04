import gc
import json
import logging
import os
from typing import Any, Dict, List

import torch
from deepeval import evaluate
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from model_utils import cleanup_model, clear_huggingface_cache
from multilingual_agent.agent import ModelNotSupported, MultilingualAgent

LANGUAGES = ["eng"]
MODELS = {
    "HuggingFaceTB/SmolLM3-3B": {},
    "Qwen/Qwen3-4B": {},
    "Qwen/Qwen3-8B": {},
    "Qwen/Qwen3-14B": {},
    # "mistralai/Mistral-Nemo-Instruct-2407": {},
    # "openai/gpt-oss-20b": {},
}


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
                            input_parameters=tool_call["arguments"],
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
                    if (
                        "function" in tool_call
                        and "name" in tool_call["function"]
                        and "arguments" in tool_call["function"]
                    ):
                        tools.append(
                            ToolCall(
                                name=tool_call["function"]["name"],
                                input_parameters=tool_call["function"]["arguments"],
                            )
                        )
                    else:
                        print(f"Wrong tool_call format: {tool_call}")

        return tools

    def evaluate_conversation(
        self, eval_item: Dict[str, Any]
    ) -> tuple[LLMTestCase, List[Dict[str, Any]]]:
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

        return test_case, agent_conversation

    def run_evaluation(self, data_path: str) -> EvaluationResult:
        """Run full evaluation on the dataset."""
        print(f"Loading evaluation data from {data_path}")
        eval_data = self.load_evaluation_data(data_path)

        test_cases = []
        agent_conversations = []
        eval_item_ids = []
        skipped = 0

        conversations_count = len(eval_data["conversations"])
        print(f"Processing {conversations_count} conversations...")

        for i, eval_item in enumerate(eval_data["conversations"]):
            progress = f"{i + 1}/{conversations_count}"
            print(f"Processing conversation {eval_item['id']} ({progress})")

            # Skip conversations that don't require tool use
            if not eval_item["requires_tool_use"]:
                print("  Skipped (no tools required)")
                skipped += 1
                continue

            test_case, agent_conversation = self.evaluate_conversation(eval_item)
            test_cases.append(test_case)
            agent_conversations.append(agent_conversation)
            eval_item_ids.append(eval_item["id"])
            print("  Processed (with tools)")

        print(
            f"\nEvaluating {len(test_cases)} test cases with Tool Correctness metric..."
        )
        print(f"Skipped {skipped} conversations without tool calls")

        # Run evaluation
        result = evaluate(test_cases, [self.tool_correctness_metric])

        # Save results to JSON file
        self._save_results(
            result, eval_data, data_path, agent_conversations, eval_item_ids
        )

        return result

    def _save_results(
        self,
        result: EvaluationResult,
        eval_data: Dict[str, Any],
        data_path: str,
        agent_conversations: List[List[Dict[str, Any]]],
        eval_item_ids: List[str],
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
            "evaluation_results": self._serialize_evaluation_result(
                result, agent_conversations, eval_item_ids
            ),
        }

        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to: {filepath}")

    def _serialize_evaluation_result(
        self,
        result: EvaluationResult,
        agent_conversations: List[List[Dict[str, Any]]],
        eval_item_ids: List[str],
    ) -> Dict[str, Any]:
        """Convert EvaluationResult to JSON-serializable dictionary."""
        # Calculate global metrics
        total_tests = len(result.test_results)
        passed_tests = sum(1 for tr in result.test_results if tr.success)
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        # Calculate average scores by metric
        metric_scores = {}
        metric_pass_rates = {}
        for test_result in result.test_results:
            for metric_data in test_result.metrics_data:
                metric_name = metric_data.name
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                    metric_pass_rates[metric_name] = []
                metric_scores[metric_name].append(metric_data.score)
                metric_pass_rates[metric_name].append(metric_data.success)

        average_scores = {}
        metric_pass_rate_percentages = {}
        for metric_name in metric_scores:
            scores = metric_scores[metric_name]
            passes = metric_pass_rates[metric_name]
            average_scores[metric_name] = sum(scores) / len(scores) if scores else 0
            metric_pass_rate_percentages[metric_name] = (
                (sum(passes) / len(passes)) * 100 if passes else 0
            )

        return {
            "global_metrics": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "overall_pass_rate": round(pass_rate, 2),
                "average_scores_by_metric": {
                    name: round(score, 4) for name, score in average_scores.items()
                },
                "pass_rates_by_metric": {
                    name: round(rate, 2)
                    for name, rate in metric_pass_rate_percentages.items()
                },
            },
            "test_results": [
                {
                    "name": test_result.name,
                    "success": test_result.success,
                    "eval_item_id": eval_item_ids[i]
                    if i < len(eval_item_ids)
                    else None,
                    "agent_conversation": (
                        agent_conversations[i] if i < len(agent_conversations) else None
                    ),
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
                for i, test_result in enumerate(result.test_results)
            ],
        }


def results_exist(model_name: str, language: str) -> bool:
    """Check if results file already exists for a model and language combination."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_name_clean = model_name.split("/")[-1]
    output_file_name = os.path.join(
        script_dir, f"results/agent_{model_name_clean}_{language}_Latn.json"
    )
    return os.path.exists(output_file_name)


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Use default data path with script directory as prefix
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "tools_use_eng.json")

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return

    current_model_name = ""
    evaluator = None

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

            # Check if results already exist
            if results_exist(actual_model_name, language):
                logger.info(
                    f"Skipping model {model_name} with language {language}, "
                    "results file exists"
                )
                continue

            # Load model only if needed and different from current
            if current_model_name != actual_model_name:
                if evaluator is not None:
                    # Clean up previous model
                    cleanup_model(evaluator.agent.model, evaluator.agent.tokenizer)
                    clear_huggingface_cache()

                # Create new evaluator
                try:
                    evaluator = AgentEvaluator(actual_model_name)
                    current_model_name = actual_model_name
                except ModelNotSupported:
                    logger.warning(
                        f"Skipping evaluation for unsupported model: "
                        f"{actual_model_name}"
                    )
                    continue

            logger.info(f"Evaluating model {model_name} with language {language}")

            # Run evaluation
            try:
                evaluator.run_evaluation(data_path)
            except Exception as e:
                logger.error(f"Error evaluating {model_name} with {language}: {e}")
                continue

            # Clear model cache after each evaluation
            if hasattr(evaluator.agent.model, "past_key_values"):
                evaluator.agent.model.past_key_values = None
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Final cleanup
    if evaluator is not None:
        cleanup_model(evaluator.agent.model, evaluator.agent.tokenizer)
        clear_huggingface_cache()


if __name__ == "__main__":
    main()
