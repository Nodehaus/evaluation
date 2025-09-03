import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent import MultilingualAgent
from deepeval import evaluate
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall


class AgentEvaluator:
    """Evaluator for the multilingual agent using DeepEval metrics."""

    def __init__(
        self, host: str = "localhost", port: int = 11434, model: str = "gemma3:1b"
    ):
        self.agent = MultilingualAgent(host=host, port=port, model=model)
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

    def evaluate_conversation(self, eval_item: Dict[str, Any]) -> Optional[LLMTestCase]:
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

    def run_evaluation(
        self, data_path: str, output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run full evaluation on the dataset."""
        print(f"Loading evaluation data from {data_path}")
        eval_data = self.load_evaluation_data(data_path)

        test_cases = []
        skipped = 0

        print(f"Processing {len(eval_data['conversations'])} conversations...")

        for i, eval_item in enumerate(eval_data["conversations"]):
            print(
                f"Processing conversation {eval_item['id']} ({i + 1}/{len(eval_data['conversations'])})"
            )

            test_case = self.evaluate_conversation(eval_item)
            if test_case:
                test_cases.append(test_case)
                tool_status = (
                    "with tools" if eval_item["requires_tool_use"] else "no tools"
                )
                print(f"  Processed ({tool_status})")
            else:
                skipped += 1
                print("  Skipped (error)")

        print(
            f"\nEvaluating {len(test_cases)} test cases with Tool Correctness metric..."
        )
        print(f"Skipped {skipped} conversations")

        if not test_cases:
            print("No test cases to evaluate!")
            return {"error": "No test cases generated"}

        # Run evaluation
        results = evaluate(test_cases, [self.tool_correctness_metric])
        print(results)

        # # Calculate summary statistics
        # passed = sum(
        #     1 for tc in test_cases if self.tool_correctness_metric.is_successful()
        # )
        # total = len(test_cases)
        # success_rate = passed / total if total > 0 else 0

        # tool_requiring_conversations = sum(
        #     1 for conv in eval_data["conversations"] if conv["requires_tool_use"]
        # )

        # evaluation_results = {
        #     "timestamp": datetime.now().isoformat(),
        #     "dataset": eval_data["dataset"],
        #     "model": self.agent.model,
        #     "total_conversations": len(eval_data["conversations"]),
        #     "tool_requiring_conversations": tool_requiring_conversations,
        #     "evaluated_conversations": total,
        #     "skipped_conversations": skipped,
        #     "tool_correctness_passed": passed,
        #     "tool_correctness_total": total,
        #     "tool_correctness_success_rate": success_rate,
        #     "individual_results": [],
        # }

        # # Add individual results
        # for i, test_case in enumerate(test_cases):
        #     # Convert ToolCall objects to dictionaries for JSON serialization
        #     tools_called = [
        #         {"name": tc.name, "parameters": tc.input_parameters}
        #         for tc in (test_case.tools_called or [])
        #     ]
        #     expected_tools = [
        #         {"name": tc.name, "parameters": tc.input_parameters}
        #         for tc in (test_case.expected_tools or [])
        #     ]

        #     result = {
        #         "test_case_index": i,
        #         "input": test_case.input,
        #         "actual_output": test_case.actual_output,
        #         "tools_called": tools_called,
        #         "expected_tools": expected_tools,
        #         "passed": self.tool_correctness_metric.is_successful(),
        #         "score": getattr(self.tool_correctness_metric, "score", None),
        #     }
        #     evaluation_results["individual_results"].append(result)

        # # Save results if output path provided
        # if output_path:
        #     with open(output_path, "w", encoding="utf-8") as f:
        #         json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        #     print(f"Results saved to {output_path}")

        return evaluation_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Multilingual Agent Tool Use")
    parser.add_argument(
        "--data",
        default="data/tools_use_eng.json",
        help="Path to evaluation data JSON file",
    )
    parser.add_argument("--host", default="localhost", help="Ollama host")
    parser.add_argument("--port", type=int, default=11434, help="Ollama port")
    parser.add_argument("--model", default="gemma3:1b", help="Ollama model name")
    parser.add_argument("--output", help="Path to save evaluation results JSON file")

    args = parser.parse_args()

    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        return

    # Create evaluator
    evaluator = AgentEvaluator(host=args.host, port=args.port, model=args.model)

    # Run evaluation
    results = evaluator.run_evaluation(args.data, args.output)

    # Print summary
    # if "error" in results:
    #     print(f"\nEvaluation failed: {results['error']}")
    # else:
    #     print(f"\n=== Evaluation Summary ===")
    #     print(f"Model: {results['model']}")
    #     print(f"Total conversations: {results['total_conversations']}")
    #     print(
    #         f"Tool-requiring conversations: {results['tool_requiring_conversations']}"
    #     )
    #     print(f"Evaluated conversations: {results['evaluated_conversations']}")
    #     print(
    #         f"Tool Correctness Success Rate: {results['tool_correctness_success_rate']:.2%} ({results['tool_correctness_passed']}/{results['tool_correctness_total']})"
    #     )

    #     if results.get("individual_results"):
    #         print(f"\nIndividual Results:")
    #         for result in results["individual_results"]:
    #             status = "✓" if result["passed"] else "✗"
    #             print(
    #                 f"  {status} Test {result['test_case_index']}: {result['input'][:50]}..."
    #             )


if __name__ == "__main__":
    main()
