import json
import random
import re
from datetime import datetime
from typing import Any, Dict, List

from model_utils import (
    chat_responses,
    check_tool_calling_support,
    load_model_and_tokenizer,
)


def weather_forecast(city: str, date: str) -> Dict[str, int]:
    """
    Get weather forecast for a specific city and date.

    Args:
        city: Name of the city
        date: Date in YYYY-MM-DD format

    Returns:
        Dictionary with temperature (Celsius), humidity (%), and wind_speed (km/h) as
        integers
    """
    # Generate realistic weather data based on city and season
    weather_patterns = {
        "New York City": {"base_temp": 20, "temp_range": 15, "humidity_base": 65},
        "London": {"base_temp": 15, "temp_range": 10, "humidity_base": 80},
        "San Francisco": {"base_temp": 18, "temp_range": 8, "humidity_base": 70},
        "Denver": {"base_temp": 22, "temp_range": 20, "humidity_base": 45},
        "Seattle": {"base_temp": 16, "temp_range": 12, "humidity_base": 85},
        "Miami": {"base_temp": 28, "temp_range": 8, "humidity_base": 75},
        "Chicago": {"base_temp": 18, "temp_range": 18, "humidity_base": 70},
    }

    # Default pattern for unknown cities
    pattern = weather_patterns.get(
        city, {"base_temp": 20, "temp_range": 15, "humidity_base": 65}
    )

    # Add some randomness
    temperature = pattern["base_temp"] + random.randint(
        -pattern["temp_range"] // 2, pattern["temp_range"] // 2
    )
    humidity = pattern["humidity_base"] + random.randint(-15, 15)
    wind_speed = random.randint(5, 35)

    # Ensure reasonable bounds
    temperature = max(-10, min(45, temperature))
    humidity = max(30, min(95, humidity))
    wind_speed = max(0, min(50, wind_speed))

    return {
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
    }


def get_current_date() -> str:
    """
    Get the current date.

    Returns:
        Current date in YYYY-MM-DD format
    """
    return datetime.now().strftime("%Y-%m-%d")


class MultilingualAgent:
    """Agent that can use tools and respond to user queries."""

    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM3-3B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        # Load model and tokenizer first
        self._load_model()

        # Check tool calling support after loading tokenizer
        self.supports_tools = check_tool_calling_support(self.tokenizer)
        if not self.supports_tools:
            raise ValueError(f"Error: Model {model_name} does not support tool calling")

    def _load_model(self):
        """Load the model and tokenizer."""
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_name)

    def _extract_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from model response text."""

        tool_calls = []

        # Pattern to match <tool_call>...</tool_call>
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        for i, match in enumerate(matches):
            try:
                # Parse the JSON inside tool_call tags
                tool_data = json.loads(match)

                # Convert to HuggingFace format
                tool_call = {
                    "type": "function",
                    "function": {
                        "name": tool_data.get("name", ""),
                        "arguments": tool_data.get("arguments", {}),
                    },
                    "id": f"call_{i + 1}",
                }
                tool_calls.append(tool_call)

            except json.JSONDecodeError:
                # Skip invalid JSON
                continue

        return tool_calls

    def chat(self, user_message: str) -> List[Dict[str, Any]]:
        """
        Process user message and return full conversation.

        Returns:
            List of conversation turns with roles: 'user', 'assistant', 'tool'
        """
        # Create initial conversation with user message
        conversation = [
            {
                "role": "system",
                "content": "You are a bot that responds to weather queries.",
            },
            {"role": "user", "content": user_message},
        ]

        # Pass the Python functions directly to chat template (if supported)
        tools = [weather_forecast, get_current_date] if self.supports_tools else None

        # Handle multiple rounds of tool calls
        max_tool_rounds = 5  # Prevent infinite loops
        tool_round = 0

        while tool_round < max_tool_rounds:
            # Generate response from model
            responses = chat_responses(
                self.model, self.tokenizer, [conversation], tools
            )
            response_text = (
                responses[0]
                if responses
                else "Sorry, I could not answer your question."
            )

            # Extract tool calls from response text
            tool_calls = self._extract_tool_calls_from_text(response_text)

            if self.supports_tools and tool_calls:
                tool_round += 1

                # Add assistant message with only tool calls (no content)
                assistant_message = {"role": "assistant", "tool_calls": tool_calls}
                conversation.append(assistant_message)

                # Execute each tool call
                unknown_tool_called = False
                for tool_call in tool_calls:
                    function_info = tool_call.get("function", {})
                    function_name = function_info.get("name")
                    function_args = function_info.get("arguments", {})

                    # Execute the tool
                    if function_name == "weather_forecast":
                        result = weather_forecast(**function_args)
                    elif function_name == "get_current_date":
                        result = get_current_date()
                    else:
                        unknown_tool_called = True
                        break

                    # Add tool result to conversation
                    conversation.append({"role": "tool", "content": json.dumps(result)})

                # If unknown tool was called, return error message
                if unknown_tool_called:
                    conversation.append(
                        {
                            "role": "assistant",
                            "content": "Sorry, I cannot answer your question.",
                        }
                    )
                    break
            else:
                # No tool calls, add final response and break
                conversation.append({"role": "assistant", "content": response_text})
                break

        return conversation


def main():
    """Example usage of the agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Multilingual Agent with Tool Support")
    parser.add_argument(
        "--model", default="HuggingFaceTB/SmolLM3-3B", help="HuggingFace model name"
    )
    parser.add_argument("--message", required=True, help="User message")

    args = parser.parse_args()

    agent = MultilingualAgent(model_name=args.model)

    try:
        conversation = agent.chat(args.message)
        print("\n=== JSON Output ===")
        print(json.dumps(conversation, indent=2))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
