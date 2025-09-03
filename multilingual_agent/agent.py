"""
Multilingual Agent with Tool Support

An agent that can use tools (specifically weather forecast) with configurable Ollama models.
"""

import json
import random
from typing import Any, Dict, List, Optional

import requests


class WeatherTool:
    """Mock weather forecast tool that returns realistic weather data."""

    def __init__(self):
        self.name = "weather_forecast"
        self.description = "Returns weather forecast with temperature in Celsius, humidity as percentage, and wind speed in km/h"
        self.parameters = {
            "city": {"type": "string", "description": "Name of the city"},
            "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
        }
        self.returns = {
            "type": "object",
            "properties": {
                "temperature": {"type": "int", "description": "Temperature in Celsius"},
                "humidity": {"type": "int", "description": "Humidity percentage"},
                "wind_speed": {"type": "int", "description": "Wind speed in km/h"},
            },
        }

    def call(self, city: str, date: str) -> Dict[str, int]:
        """Simulate weather forecast API call with realistic data."""
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


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, host: str = "localhost", port: int = 11434):
        self.base_url = f"http://{host}:{port}"

    def chat(
        self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7
    ) -> str:
        """Send chat request to Ollama."""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to Ollama at {self.base_url}: {e}")


class MultilingualAgent:
    """Agent that can use tools and respond to user queries."""

    def __init__(
        self, host: str = "localhost", port: int = 11434, model: str = "gemma3:1b"
    ):
        self.ollama_client = OllamaClient(host, port)
        self.model = model
        self.tools = {"weather_forecast": WeatherTool()}

        # Generate system prompt dynamically from tool definitions
        self.system_prompt = self._generate_system_prompt()
        print(self.system_prompt)

    def _generate_system_prompt(self) -> str:
        """Generate system prompt dynamically from available tools."""
        base_prompt = """You are a helpful assistant that can use tools to answer user questions.

Available tools:"""

        # Generate tool descriptions
        for tool in self.tools.values():
            # Build parameter signature
            param_parts = []
            for param_name, param_info in tool.parameters.items():
                param_type = param_info["type"]
                param_parts.append(f"{param_name}: {param_type}")

            param_signature = ", ".join(param_parts)

            # Build return type description
            return_desc = ""
            if hasattr(tool, "returns") and tool.returns:
                if (
                    tool.returns.get("type") == "object"
                    and "properties" in tool.returns
                ):
                    props = []
                    for prop_name, prop_info in tool.returns["properties"].items():
                        prop_type = prop_info["type"]
                        props.append(f'"{prop_name}": {prop_type}')
                    return_desc = "{" + ", ".join(props) + "}"
                else:
                    return_desc = str(tool.returns.get("type", "unknown"))

            # Add tool description line
            tool_line = f"- {tool.name}({param_signature})"
            if return_desc:
                tool_line += f" -> {return_desc}"
            base_prompt += f"\n{tool_line}"
            base_prompt += f"\n  {tool.description}"

            # Add parameter descriptions
            if tool.parameters:
                base_prompt += "\n  Parameters:"
                for param_name, param_info in tool.parameters.items():
                    param_desc = param_info.get("description", "")
                    base_prompt += f"\n    - {param_name}: {param_desc}"

        # Add usage instructions
        usage_instructions = """

When you need to use a tool:
1. Decide if a tool use is necessary for the user question.
2. If yes, then you must return an output in the format: TOOL_CALL: tool_name(param1="value1", param2="value2")
3. Wait for the tool result
4. Interpret the results and provide a helpful response

If you don't need a tool, just answer the question directly.

Be conversational and helpful. Always explain the weather data in a user-friendly way."""

        return base_prompt + usage_instructions

    def _parse_tool_call(self, text: str) -> Optional[Dict]:
        """Parse tool call from agent response."""
        if "TOOL_CALL:" not in text:
            return None

        try:
            # Extract tool call line
            lines = text.split("\n")
            tool_line = None
            for line in lines:
                if "TOOL_CALL:" in line:
                    tool_line = line.split("TOOL_CALL:")[1].strip()
                    break

            if not tool_line:
                return None

            print(tool_line)

            # Parse tool name and parameters
            if "(" not in tool_line or ")" not in tool_line:
                return None

            tool_name = tool_line.split("(")[0].strip()
            params_str = tool_line.split("(")[1].split(")")[0]

            # Simple parameter parsing (assumes string parameters with quotes)
            params = {}
            if params_str.strip():
                for param in params_str.split(","):
                    if "=" in param:
                        key, value = param.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("\"'")
                        params[key] = value

            return {"tool": tool_name, "parameters": params}

        except Exception:
            return None

    def chat(self, user_message: str) -> List[Dict[str, Any]]:
        """
        Process user message and return full conversation.

        Returns:
            List of conversation turns with roles: 'user', 'assistant', 'tool'
        """
        conversation = [{"role": "user", "content": user_message}]

        # Prepare messages for Ollama
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Get initial response
        response = self.ollama_client.chat(self.model, messages)

        # Check if response contains tool call
        tool_call = self._parse_tool_call(response)

        if tool_call and tool_call["tool"] in self.tools:
            # Validate that required parameters are available
            required_params = ["city", "date"]
            tool_params = tool_call["parameters"]

            if not all(
                param in tool_params and tool_params[param] for param in required_params
            ):
                # Missing required parameters, return error response
                conversation.append(
                    {
                        "role": "assistant",
                        "content": "Sorry, I could not answer your question.",
                    }
                )
                return conversation

            # Add assistant message with tool call intention
            conversation.append(
                {
                    "role": "assistant",
                    "content": response,
                    "tool_calls": [
                        {
                            "name": tool_call["tool"],
                            "parameters": tool_call["parameters"],
                        }
                    ],
                }
            )

            # Execute tool
            tool = self.tools[tool_call["tool"]]
            tool_result = tool.call(**tool_call["parameters"])

            # Add tool result to conversation
            conversation.append({"role": "tool", "content": json.dumps(tool_result)})

            # Get final response with tool result
            messages.extend(
                [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": f"Tool result: {json.dumps(tool_result)}. "
                        f"Please interpret this result and provide a helpful response to the user.",
                    },
                ]
            )

            final_response = self.ollama_client.chat(self.model, messages)
            conversation.append({"role": "assistant", "content": final_response})
        else:
            # No tool needed, just add the response
            conversation.append({"role": "assistant", "content": response})

        return conversation


def main():
    """Example usage of the agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Multilingual Agent with Tool Support")
    parser.add_argument("--host", default="localhost", help="Ollama host")
    parser.add_argument("--port", type=int, default=11434, help="Ollama port")
    parser.add_argument("--model", default="gemma3:1b", help="Ollama model name")
    parser.add_argument("--message", required=True, help="User message")

    args = parser.parse_args()

    agent = MultilingualAgent(host=args.host, port=args.port, model=args.model)

    try:
        conversation = agent.chat(args.message)

        print("=== Conversation ===")
        for turn in conversation:
            role = turn["role"].upper()
            content = turn["content"]

            if "tool_calls" in turn:
                tool_info = turn["tool_calls"][0]
                print(f"\n{role}: {content}")
                print(f"[TOOL CALL: {tool_info['name']}({tool_info['parameters']})]")
            else:
                print(f"\n{role}: {content}")

        print("\n=== JSON Output ===")
        print(json.dumps(conversation, indent=2))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
