"""
modules/08_exceptions/05_model_behavior_error.py

ModelBehaviorError Exception Demonstration

Description:
    Demonstrates how to handle and report ModelBehaviorError exceptions in the OpenAI Agents SDK. Intentionally triggers a ModelBehaviorError by instructing the agent to call a non-existent tool, showing how to catch and report SDK exceptions in a robust, educational, and production-ready way.

Features:
    - Demonstrates robust exception handling for ModelBehaviorError
    - Visually distinct, clearly labeled output with banners and emoji
    - Educational comments and section headers
    - Explains intentional ModelBehaviorError for demonstration

Environment Variables:
    - MISTRAL_API_KEY: API key for Mistral model (required)

Author:
    Zohaib Khan

References:
    - https://github.com/openai/agents-sdk
    - https://platform.openai.com/docs/agents
    - https://github.com/openai/openai-python
"""
import asyncio
import os
from agents import (
    Agent,
    Runner,
    function_tool,
    exceptions,
    set_tracing_disabled,
)
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import find_dotenv, load_dotenv

# =========================
# 1. Environment & Model Setup
# =========================

load_dotenv(find_dotenv())
# set_tracing_disabled(True)

MISTRAL_API_KEY: str | None = os.getenv("MISTRAL_API_KEY")
MODEL_NAME: str = "mistral/mistral-small-2506"

if not MISTRAL_API_KEY:
    print("\n❌ [ERROR] MISTRAL_API_KEY environment variable is required but not found.\n")
    raise ValueError("MISTRAL_API_KEY environment variable is required but not found.")

print("\n" + "="*60)
print(f"🚀 Initializing OpenRouter client with model: {MODEL_NAME}")
model = LitellmModel(model=MODEL_NAME, api_key=MISTRAL_API_KEY)
print(f"✅ Model configured successfully")
print("="*60 + "\n")

# =========================
# 2. ModelBehaviorError Exception Demo
# =========================

@function_tool
def get_current_weather(location: str) -> str:
    """
    Gets the current weather for a given location.

    Args:
        location (str): The location to get weather for.
    Returns:
        str: Weather information for the location.
    """
    print(f"[TOOL] get_current_weather called for {location}")
    if location == "testville":
        return "The weather in Testville is sunny."
    return f"Weather data not available for {location}."

async def demo_model_behavior_error():
    """
    Demonstrates how the SDK raises and handles a ModelBehaviorError when the agent is instructed to call a non-existent tool.
    This intentionally triggers the error for educational purposes.
    """
    print("\n==============================")
    print("🧪 [DEMO] Exception Handling: ModelBehaviorError (Non-existent Tool)")
    print("==============================\n")
    try:
        # Intentionally instruct the agent to call a tool that does not exist
        print("📝 [INPUT] Agent instructed to call a non-existent tool 'get_stock_price' (intentional error)\n")
        agent = Agent(
            name="MisconfiguredAgent",
            instructions=(
                "You are a helpful assistant. "
                "VERY IMPORTANT: You MUST try to call a tool named 'get_stock_price' with any argument, even though it does not exist."
            ),
            tools=[get_current_weather]
        )
        response = await Runner.run(agent, "What is the weather and stock price?")
        print(f"🤖 [OUTPUT] Runner result: {response.final_output}")
    except exceptions.ModelBehaviorError as e:
        print(f"\n❗ [EXCEPTION] Caught ModelBehaviorError as expected: {e}\n")
    except Exception as e:
        print(f"\n❌ [ERROR] Unexpected error in demo_model_behavior_error: {type(e).__name__} - {e}\n")
    else:
        print("\n⚠️ [WARNING] No exception was raised, which is unexpected for this demo.\n")

# =========================
# 3. Main Entrypoint
# =========================

async def main():
    """
    Run the ModelBehaviorError exception handling demonstration.
    """
    print("\n==================================================")
    print("🧪 Running ModelBehaviorError Exception Handling Demo")
    print("==================================================\n")
    await demo_model_behavior_error()
    print("\n==================================================")
    print("✅ [COMPLETE] ModelBehaviorError exception handling demo finished!")
    print("==================================================\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ [ERROR] Unhandled exception in main: {e}\n")
