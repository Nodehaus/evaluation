import json
import re
from datetime import datetime
from typing import Any, Dict, List

from model_utils import (
    chat_responses,
    check_tool_calling_support,
    load_model_and_tokenizer,
)


class ModelNotSupported(Exception):
    """Exception raised when a model does not support tool calling."""

    pass


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
        # North America
        "New York City": {
            "base_temp": 20,
            "temp_range": 15,
            "humidity_base": 65,
            "hemisphere": "northern",
        },
        "San Francisco": {
            "base_temp": 18,
            "temp_range": 8,
            "humidity_base": 70,
            "hemisphere": "northern",
        },
        "Denver": {
            "base_temp": 22,
            "temp_range": 20,
            "humidity_base": 45,
            "hemisphere": "northern",
        },
        "Seattle": {
            "base_temp": 16,
            "temp_range": 12,
            "humidity_base": 85,
            "hemisphere": "northern",
        },
        "Miami": {
            "base_temp": 28,
            "temp_range": 8,
            "humidity_base": 75,
            "hemisphere": "northern",
        },
        "Chicago": {
            "base_temp": 18,
            "temp_range": 18,
            "humidity_base": 70,
            "hemisphere": "northern",
        },
        # Europe
        "London": {
            "base_temp": 15,
            "temp_range": 10,
            "humidity_base": 80,
            "hemisphere": "northern",
        },
        "Paris": {
            "base_temp": 16,
            "temp_range": 12,
            "humidity_base": 75,
            "hemisphere": "northern",
        },
        "Berlin": {
            "base_temp": 14,
            "temp_range": 14,
            "humidity_base": 70,
            "hemisphere": "northern",
        },
        "Rome": {
            "base_temp": 21,
            "temp_range": 10,
            "humidity_base": 65,
            "hemisphere": "northern",
        },
        "Madrid": {
            "base_temp": 19,
            "temp_range": 12,
            "humidity_base": 55,
            "hemisphere": "northern",
        },
        "Amsterdam": {
            "base_temp": 13,
            "temp_range": 8,
            "humidity_base": 85,
            "hemisphere": "northern",
        },
        "Stockholm": {
            "base_temp": 9,
            "temp_range": 16,
            "humidity_base": 75,
            "hemisphere": "northern",
        },
        # Asia
        "Tokyo": {
            "base_temp": 18,
            "temp_range": 14,
            "humidity_base": 70,
            "hemisphere": "northern",
        },
        "Beijing": {
            "base_temp": 17,
            "temp_range": 18,
            "humidity_base": 60,
            "hemisphere": "northern",
        },
        "Mumbai": {
            "base_temp": 30,
            "temp_range": 6,
            "humidity_base": 80,
            "hemisphere": "northern",
        },
        "Bangkok": {
            "base_temp": 32,
            "temp_range": 4,
            "humidity_base": 75,
            "hemisphere": "northern",
        },
        "Seoul": {
            "base_temp": 16,
            "temp_range": 16,
            "humidity_base": 65,
            "hemisphere": "northern",
        },
        "Singapore": {
            "base_temp": 30,
            "temp_range": 3,
            "humidity_base": 85,
            "hemisphere": "northern",
        },
        "Dubai": {
            "base_temp": 32,
            "temp_range": 8,
            "humidity_base": 45,
            "hemisphere": "northern",
        },
        # South America
        "São Paulo": {
            "base_temp": 21,
            "temp_range": 8,
            "humidity_base": 75,
            "hemisphere": "southern",
        },
        "Buenos Aires": {
            "base_temp": 18,
            "temp_range": 10,
            "humidity_base": 70,
            "hemisphere": "southern",
        },
        "Rio de Janeiro": {
            "base_temp": 26,
            "temp_range": 6,
            "humidity_base": 80,
            "hemisphere": "southern",
        },
        "Lima": {
            "base_temp": 19,
            "temp_range": 6,
            "humidity_base": 85,
            "hemisphere": "southern",
        },
        "Bogotá": {
            "base_temp": 15,
            "temp_range": 4,
            "humidity_base": 75,
            "hemisphere": "northern",
        },
        "Santiago": {
            "base_temp": 16,
            "temp_range": 12,
            "humidity_base": 60,
            "hemisphere": "southern",
        },
        "Caracas": {
            "base_temp": 25,
            "temp_range": 4,
            "humidity_base": 70,
            "hemisphere": "northern",
        },
        # Africa
        "Cairo": {
            "base_temp": 28,
            "temp_range": 10,
            "humidity_base": 40,
            "hemisphere": "northern",
        },
        "Lagos": {
            "base_temp": 29,
            "temp_range": 4,
            "humidity_base": 85,
            "hemisphere": "northern",
        },
        "Cape Town": {
            "base_temp": 20,
            "temp_range": 8,
            "humidity_base": 65,
            "hemisphere": "southern",
        },
        "Nairobi": {
            "base_temp": 22,
            "temp_range": 6,
            "humidity_base": 60,
            "hemisphere": "southern",
        },
        "Casablanca": {
            "base_temp": 21,
            "temp_range": 8,
            "humidity_base": 70,
            "hemisphere": "northern",
        },
        "Addis Ababa": {
            "base_temp": 18,
            "temp_range": 6,
            "humidity_base": 55,
            "hemisphere": "northern",
        },
        "Johannesburg": {
            "base_temp": 19,
            "temp_range": 10,
            "humidity_base": 50,
            "hemisphere": "southern",
        },
    }

    # Default pattern for unknown cities
    pattern = weather_patterns.get(
        city,
        {
            "base_temp": 20,
            "temp_range": 15,
            "humidity_base": 65,
            "hemisphere": "northern",
        },
    )

    # Parse date to determine season for seasonal adjustments
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        month = date_obj.month
    except (ValueError, TypeError):
        month = datetime.now().month  # fallback to current month

    # Determine season (Northern Hemisphere as default)
    # For Southern Hemisphere cities, we'll flip the seasons
    is_southern = pattern.get("hemisphere") == "southern"

    if is_southern:
        # Flip seasons for Southern Hemisphere
        if month in [12, 1, 2]:  # Summer in Southern Hemisphere
            season_temp_modifier = pattern["temp_range"] // 3
            season_humidity_modifier = 10
        elif month in [3, 4, 5]:  # Autumn in Southern Hemisphere
            season_temp_modifier = 0
            season_humidity_modifier = 0
        elif month in [6, 7, 8]:  # Winter in Southern Hemisphere
            season_temp_modifier = -pattern["temp_range"] // 2
            season_humidity_modifier = -10
        else:  # Spring in Southern Hemisphere (9, 10, 11)
            season_temp_modifier = pattern["temp_range"] // 4
            season_humidity_modifier = 5
    else:
        # Northern Hemisphere seasons
        if month in [12, 1, 2]:  # Winter
            season_temp_modifier = -pattern["temp_range"] // 2
            season_humidity_modifier = -10
        elif month in [3, 4, 5]:  # Spring
            season_temp_modifier = pattern["temp_range"] // 4
            season_humidity_modifier = 5
        elif month in [6, 7, 8]:  # Summer
            season_temp_modifier = pattern["temp_range"] // 3
            season_humidity_modifier = 10
        else:  # Autumn (9, 10, 11)
            season_temp_modifier = 0
            season_humidity_modifier = 0

    # Apply seasonal adjustments
    temperature = pattern["base_temp"] + season_temp_modifier
    humidity = pattern["humidity_base"] + season_humidity_modifier

    # Wind speed based on season (higher in winter/stormy seasons)
    if month in [11, 12, 1, 2, 3]:  # Storm season
        wind_speed = 15 + (pattern["temp_range"] // 3)
    else:
        wind_speed = 10 + (pattern["temp_range"] // 4)

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
    return "2025-09-11"  # Day of the presentation


class MultilingualAgent:
    """Agent that can use tools and respond to user queries in multiple languages."""

    def __init__(
        self, model_name: str = "HuggingFaceTB/SmolLM3-3B", language: str = "eng"
    ):
        self.model_name = model_name
        self.language = language
        self.model = None
        self.tokenizer = None

        # Load model and tokenizer first
        self._load_model()

        # Check tool calling support after loading tokenizer
        if not check_tool_calling_support(self.tokenizer):
            raise ModelNotSupported(f"Model {model_name} does not support tool calling")

        # Initialize translations
        self._init_translations()

    def _init_translations(self):
        """Initialize translation dictionaries for system prompts and error messages."""
        self.system_prompts = {
            "eng": (
                "You are a bot that responds to weather queries. Your final answer "
                "to the user question must be in natural language."
            ),
            "deu": (
                "Du bist ein Bot, der auf Wetteranfragen antwortet. Benutze für Städtenamen die englische Form des Namens. Deine endgültige "
                "Antwort auf die Benutzerfrage muss in natürlicher Sprache erfolgen. "
            ),
            "fra": (
                "Vous êtes un bot qui répond aux requêtes météorologiques. "
                "Utilisez la forme anglaise des noms de villes. Votre "
                "réponse finale à la question de l'utilisateur doit être en "
                "langage naturel."
            ),
            "por": (
                "Você é um bot que responde a consultas meteorológicas. "
                "Use a forma inglesa dos nomes das cidades. Sua resposta "
                "final à pergunta do usuário deve ser em linguagem natural."
            ),
            "nld": (
                "U bent een bot die reageert op weervragen. "
                "Gebruik de Engelse vorm van stadsnamen. Uw eindantwoord op de "
                "vraag van de gebruiker moet in natuurlijke taal zijn."
            ),
            "pol": (
                "Jesteś botem, który odpowiada na zapytania pogodowe. "
                "Używaj angielskiej formy nazw miast. Twoja ostateczna "
                "odpowiedź na pytanie użytkownika musi być w języku naturalnym."
            ),
            "est": (
                "Olete bot, mis vastab ilmapäringutele. "
                "Kasutage linnanimetuste inglisekeelset vormi. Teie lõplik vastus "
                "kasutaja küsimusele peab olema loomulik keel."
            ),
        }

        self.error_messages = {
            "eng": {
                "cannot_answer": "Sorry, I could not answer your question.",
                "unknown_tool": "Sorry, I cannot answer your question.",
            },
            "deu": {
                "cannot_answer": (
                    "Entschuldigung, ich konnte Ihre Frage nicht beantworten."
                ),
                "unknown_tool": (
                    "Entschuldigung, ich kann Ihre Frage nicht beantworten."
                ),
            },
            "fra": {
                "cannot_answer": "Désolé, je n'ai pas pu répondre à votre question.",
                "unknown_tool": "Désolé, je ne peux pas répondre à votre question.",
            },
            "por": {
                "cannot_answer": "Desculpe, não consegui responder à sua pergunta.",
                "unknown_tool": "Desculpe, não posso responder à sua pergunta.",
            },
            "nld": {
                "cannot_answer": "Sorry, ik kon uw vraag niet beantwoorden.",
                "unknown_tool": "Sorry, ik kan uw vraag niet beantwoorden.",
            },
            "pol": {
                "cannot_answer": (
                    "Przepraszam, nie mogłem odpowiedzieć na Twoje pytanie."
                ),
                "unknown_tool": (
                    "Przepraszam, nie mogę odpowiedzieć na Twoje pytanie."
                ),
            },
            "est": {
                "cannot_answer": ("Vabandust, ma ei saanud teie küsimusele vastata."),
                "unknown_tool": "Vabandust, ma ei saa teie küsimusele vastata.",
            },
        }

    def _load_model(self):
        """Load the model and tokenizer."""
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_name)

    def _get_tool_call_pattern_from_template(self) -> str:
        """Extract the tool call pattern from the chat template."""
        if not hasattr(self.tokenizer, "chat_template"):
            return r"<tool_call>\s*(\{.*?\})\s*</tool_call>"  # Default fallback

        template = self.tokenizer.chat_template.lower()

        # Check for different tool call formats based on template content
        if "<tool_call>" in template and "</tool_call>" in template:
            # SmolLM3 format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
            return r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        elif "<function_call>" in template and "</function_call>" in template:
            # Some models might use different XML-like tags
            return r"<function_call>\s*(\{.*?\})\s*</function_call>"
        elif "[tool_calls]" in template or "tool_calls" in template:
            # Pattern for: [TOOL_CALLS][{"name": "func", "arguments": {}}]</s>
            return r"\[TOOL_CALLS\]\[(\{.*?\})\]"
        elif "<|channel|>commentary" in template and "<|call|>" in template:
            # Pattern: <|channel|>commentary to=func_name <|message|>{...}<|call|>
            return r"<\|channel\|>commentary to=([^<\s]+).*?<\|message\|>(\{.*?\})<\|call\|>"
        else:
            # Default pattern
            return r"<tool_call>\s*(\{.*?\})\s*</tool_call>"

    def _extract_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from model response using template-derived pattern."""

        tool_calls = []

        # Get pattern from chat template
        pattern = self._get_tool_call_pattern_from_template()
        matches = re.findall(pattern, text, re.DOTALL)

        for i, match in enumerate(matches):
            try:
                # Handle different pattern formats
                if r"<\|channel\|>commentary" in pattern:
                    # Commentary pattern: (func_name, json_args)
                    if isinstance(match, tuple) and len(match) == 2:
                        func_name, json_args = match
                        # Extract function name after "functions."
                        if func_name.startswith("functions."):
                            func_name = func_name[10:]  # Remove "functions." prefix
                        tool_data = json.loads(json_args)
                        tool_call = {
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": tool_data,
                            },
                            "id": f"call_{i + 1}",
                        }
                    else:
                        continue
                else:
                    # Standard patterns: JSON with name and arguments
                    tool_data = json.loads(match)
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
                "content": self.system_prompts.get(self.language),
            },
            {"role": "user", "content": user_message},
        ]

        # Pass the Python functions directly to chat template (if supported)
        tools = [weather_forecast, get_current_date]

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
                else self.error_messages.get(self.language, self.error_messages["eng"])[
                    "cannot_answer"
                ]
            )

            # Extract tool calls from response text
            tool_calls = self._extract_tool_calls_from_text(response_text)

            if tool_calls:
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
                    if isinstance(result, str):
                        conversation.append({"role": "tool", "content": result})
                    else:
                        conversation.append(
                            {"role": "tool", "content": json.dumps(result)}
                        )

                # If unknown tool was called, return error message
                if unknown_tool_called:
                    conversation.append(
                        {
                            "role": "assistant",
                            "content": self.error_messages.get(
                                self.language, self.error_messages["eng"]
                            )["unknown_tool"],
                        }
                    )
                    break
            else:
                conversation.append(
                    {"role": "assistant", "content": response_text.strip()}
                )
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
    parser.add_argument(
        "--language",
        default="eng",
        choices=["eng", "deu", "fra", "por", "nld", "pol", "est"],
        help="Language code for the agent responses",
    )

    args = parser.parse_args()

    agent = MultilingualAgent(model_name=args.model, language=args.language)

    conversation = agent.chat(args.message)
    print("\n=== JSON Output ===")
    print(json.dumps(conversation, indent=2))


if __name__ == "__main__":
    main()
