import gc
import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch

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


class ToolCall:
    """Represents a tool call for evaluation."""

    def __init__(self, name: str, arguments: Dict[str, Any]):
        self.name = name
        self.arguments = arguments

    def __eq__(self, other):
        if not isinstance(other, ToolCall):
            return False
        return self.name == other.name and self.arguments == other.arguments

    def __repr__(self):
        return f"ToolCall(name={self.name}, arguments={self.arguments})"


class MetricResult:
    """Result of a single metric evaluation."""

    def __init__(self, name: str, score: float, success: bool, reason: str = ""):
        self.name = name
        self.score = score
        self.success = success
        self.reason = reason


class TestResult:
    """Result of evaluating a single test case."""

    def __init__(
        self,
        test_id: str,
        success: bool,
        metrics: List[MetricResult],
        agent_conversation: List[Dict[str, Any]],
        eval_item_id: Optional[str] = None,
    ):
        self.test_id = test_id
        self.success = success
        self.metrics = metrics
        self.agent_conversation = agent_conversation
        self.eval_item_id = eval_item_id


class EvaluationResult:
    """Overall evaluation results."""

    def __init__(self, test_results: List[TestResult]):
        self.test_results = test_results


class AgentEvaluator:
    """Evaluator for the multilingual agent using custom metrics."""

    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM3-3B"):
        self.agent = MultilingualAgent(model_name=model_name)

    def load_evaluation_data(self, data_path: str) -> Dict[str, Any]:
        """Load evaluation dataset from JSON file."""
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def extract_expected_tools_from_conversation(
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
                            arguments=tool_call["arguments"],
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
                                arguments=tool_call["function"]["arguments"],
                            )
                        )
                    else:
                        print(f"Wrong tool_call format: {tool_call}")

        return tools

    def calculate_tool_correctness(
        self, expected_tools: List[ToolCall], actual_tools: List[ToolCall]
    ) -> MetricResult:
        """Calculate tool correctness metric."""
        if not expected_tools and not actual_tools:
            return MetricResult(
                "Tool Correctness", 1.0, True, "No tools expected or called"
            )

        if not expected_tools and actual_tools:
            return MetricResult(
                "Tool Correctness",
                0.0,
                False,
                f"No tools expected but {len(actual_tools)} were called",
            )

        if expected_tools and not actual_tools:
            return MetricResult(
                "Tool Correctness",
                0.0,
                False,
                f"Expected {len(expected_tools)} tools but none were called",
            )

        # Check tool names match
        expected_names = [tool.name for tool in expected_tools]
        actual_names = [tool.name for tool in actual_tools]

        correct_tools = 0
        total_expected = len(expected_tools)

        for expected_name in expected_names:
            if expected_name in actual_names:
                correct_tools += 1

        score = correct_tools / total_expected if total_expected > 0 else 0.0
        success = score >= 0.7  # 70% threshold

        reason = f"Called {correct_tools}/{total_expected} expected tools"
        return MetricResult("Tool Correctness", score, success, reason)

    def calculate_argument_correctness(
        self, expected_tools: List[ToolCall], actual_tools: List[ToolCall]
    ) -> MetricResult:
        """Calculate argument correctness metric."""
        if not expected_tools and not actual_tools:
            return MetricResult(
                "Argument Correctness", 1.0, True, "No arguments to check"
            )

        if not expected_tools or not actual_tools:
            return MetricResult("Argument Correctness", 0.0, False, "Tool mismatch")

        correct_args = 0
        total_tools = 0

        # Match tools by name and check arguments
        for expected_tool in expected_tools:
            matching_actual = [t for t in actual_tools if t.name == expected_tool.name]
            if matching_actual:
                total_tools += 1
                actual_tool = matching_actual[0]  # Take first match
                if expected_tool.arguments == actual_tool.arguments:
                    correct_args += 1

        score = correct_args / total_tools if total_tools > 0 else 0.0
        success = score >= 0.7  # 70% threshold

        reason = f"Correct arguments for {correct_args}/{total_tools} tools"
        return MetricResult("Argument Correctness", score, success, reason)

    def calculate_task_completion(
        self,
        user_message: str,
        agent_conversation: List[Dict[str, Any]],
        requires_tool_use: bool,
    ) -> MetricResult:
        """Calculate task completion metric."""
        # Simple heuristic: task is completed if:
        # 1. For tool-required tasks: tools were used and final response is present
        # 2. For non-tool tasks: a reasonable response was provided

        final_response = ""
        tools_used = False

        for turn in agent_conversation:
            if turn["role"] == "assistant":
                if "tool_calls" in turn:
                    tools_used = True
                if "content" in turn and turn["content"]:
                    final_response = turn["content"]

        if requires_tool_use:
            if tools_used and final_response.strip():
                score = 1.0
                reason = "Tools used and response provided"
            elif tools_used:
                score = 0.7
                reason = "Tools used but incomplete response"
            else:
                score = 0.0
                reason = "Required tools not used"
        else:
            if final_response.strip() and len(final_response.strip()) > 10:
                score = 1.0
                reason = "Reasonable response provided"
            else:
                score = 0.0
                reason = "No adequate response provided"

        success = score >= 0.7
        return MetricResult("Task Completion", score, success, reason)

    def evaluate_conversation(self, eval_item: Dict[str, Any]) -> TestResult:
        """Evaluate a single conversation item."""
        conversation = eval_item["conversation"]
        requires_tool_use = eval_item["requires_tool_use"]
        eval_item_id = eval_item.get("id", "")

        # Get the initial user message
        user_message = conversation[0]["content"]

        # Generate agent response
        agent_conversation = self.agent.chat(user_message)

        # Extract expected and actual tool calls
        expected_tools = self.extract_expected_tools_from_conversation(conversation)
        actual_tools = self.extract_actual_tools_from_conversation(agent_conversation)

        # Calculate metrics
        metrics = []

        # Tool correctness (always calculated)
        tool_correctness = self.calculate_tool_correctness(expected_tools, actual_tools)
        metrics.append(tool_correctness)

        # Argument correctness (only if tools are involved)
        if expected_tools or actual_tools:
            arg_correctness = self.calculate_argument_correctness(
                expected_tools, actual_tools
            )
            metrics.append(arg_correctness)

        # Task completion (always calculated)
        task_completion = self.calculate_task_completion(
            user_message, agent_conversation, requires_tool_use
        )
        metrics.append(task_completion)

        # Determine overall success (all metrics must pass)
        overall_success = all(metric.success for metric in metrics)

        return TestResult(
            test_id=f"test_{eval_item_id}",
            success=overall_success,
            metrics=metrics,
            agent_conversation=agent_conversation,
            eval_item_id=eval_item_id,
        )

    def run_evaluation(self, data_path: str) -> EvaluationResult:
        """Run full evaluation on the dataset."""
        print(f"Loading evaluation data from {data_path}")
        eval_data = self.load_evaluation_data(data_path)

        test_cases = []

        conversations_count = len(eval_data["conversations"])
        print(f"Processing {conversations_count} conversations...")

        for i, eval_item in enumerate(eval_data["conversations"]):
            progress = f"{i + 1}/{conversations_count}"
            print(f"Processing conversation {eval_item['id']} ({progress})")

            # Process all conversations (including those without tool use)
            test_result = self.evaluate_conversation(eval_item)
            test_cases.append(test_result)

            status = "✓ Pass" if test_result.success else "✗ Fail"
            print(f"  {status}")

        print(f"\nEvaluation completed for {len(test_cases)} conversations")

        # Create evaluation result
        result = EvaluationResult(test_cases)

        # Save results to JSON file
        self._save_results(result, eval_data, data_path)

        return result

    def _save_results(
        self,
        result: EvaluationResult,
        eval_data: Dict[str, Any],
        data_path: str,
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

    def _serialize_evaluation_result(
        self,
        result: EvaluationResult,
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
            for metric in test_result.metrics:
                metric_name = metric.name
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                    metric_pass_rates[metric_name] = []
                metric_scores[metric_name].append(metric.score)
                metric_pass_rates[metric_name].append(metric.success)

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
                    "test_id": test_result.test_id,
                    "success": test_result.success,
                    "eval_item_id": test_result.eval_item_id,
                    "agent_conversation": test_result.agent_conversation,
                    "metrics_data": [
                        {
                            "name": metric.name,
                            "success": metric.success,
                            "score": metric.score,
                            "reason": metric.reason,
                        }
                        for metric in test_result.metrics
                    ],
                }
                for test_result in result.test_results
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
