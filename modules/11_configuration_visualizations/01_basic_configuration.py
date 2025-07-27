import asyncio
from enum import Enum
import os
from pyexpat import model
from typing import List, Literal
from agents.extensions.models.litellm_model import LitellmModel
from pydantic import BaseModel
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    enable_verbose_stdout_logging,
    function_tool,
    handoff,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_export_api_key,
)
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
import logging

# =========================
# Environment & Model Setup
# =========================
load_dotenv(find_dotenv())

# Configuration constants


async def demo_basic_configuration():
    print(f"Demo : Basic SDK Configuration")

    print(f"API KEY CONFIGURATION")
    # Get key from env (should already be loaded)
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    if not GEMINI_API_KEY:
        print(
            "\n❌ [ERROR] GEMINI_API_KEY environment variable is required but not found.\n"
        )
        raise ValueError(
            "GEMINI_API_KEY environment variable is required but not found."
        )

    try:
        # Custom client setup
        custom_client = AsyncOpenAI(
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_BASE_URL,  # Can be changed to proxy or other provider
            timeout=20.0,  # Custom timeout
            max_retries=2,  # Custom retry policy
        )

        # Register it globally in the SDK
        set_default_openai_client(custom_client)

        print("✅ Custom OpenAI client set successfully!")
        print(f"   ➤ Base URL: {custom_client.base_url}")
        print(f"   ➤ Timeout: {custom_client.timeout} seconds")
        print(f"   ➤ Max retries: {custom_client.max_retries}")

        GEMINI_MODEL_NAME: str = "gemini-2.0-flash"

        print(f"\n==============================")
        print(f"🚀 Initializing Gemini client with model: {GEMINI_MODEL_NAME}")
        print(f"==============================\n")

        # Configure the chat completions model
        model = OpenAIChatCompletionsModel(
            openai_client=custom_client, model=GEMINI_MODEL_NAME
        )

        print(f"✅ [INFO] Model configured successfully\n")

    except Exception as e:
        print(f"❌ Failed to configure custom client: {e}")

    # 3. API selection (Responses API vs Chat Completions API)
    print("\nAPI Selection:")

    try:
        # By default, SDK uses OpenAI Responses API
        print("🔄 Default: Using OpenAI Responses API")

        # Switch to Chat Completions API if needed
        set_default_openai_api("chat_completions")
        print(
            "💡 Can switch to Chat Completions API with set_default_openai_api('chat_completions')"
        )

    except Exception as e:
        print(f"❌ Error configuring API selection: {e}")


async def demo_tracing_configuration():
    print("\n📊 Tracing Configuration")

    print("🔍 Tracing is enabled by default in the SDK.")

    # 1. Set tracing key
    tracing_key = os.getenv("OPENAI_API_KEY")  # Usually same as main key

    if tracing_key:
        try:
            set_tracing_export_api_key(tracing_key)
            print("✅ Tracing API key has been set.")
        except Exception as e:
            print(f"❌ Failed to set tracing key: {e}")
    else:
        print("⚠️ No tracing key found in env. You can set it using:")
        print("   ➤ set_tracing_export_api_key('your-key')")

    # 2. Optionally disable tracing
    print("\n🛑 You can disable tracing (not recommended during dev):")
    print("   ➤ set_tracing_disabled(True)")
    print("   (Keeping tracing enabled for this demo)")

    # 3. Show sensitive data logging settings
    print("\n🔐 Sensitive Data Logging Controls:")

    model_data_flag = os.getenv("OPENAI_AGENTS_DONT_LOG_MODEL_DATA")
    tool_data_flag = os.getenv("OPENAI_AGENTS_DONT_LOG_TOOL_DATA")

    print(
        f"   - Model data logging: {'❌ DISABLED' if model_data_flag else '✅ ENABLED'}"
    )
    print(
        f"   - Tool data logging:  {'❌ DISABLED' if tool_data_flag else '✅ ENABLED'}"
    )

    print("\n💡 Tip: Set these in `.env` for better privacy in development:")
    print("   OPENAI_AGENTS_DONT_LOG_MODEL_DATA=1")
    print("   OPENAI_AGENTS_DONT_LOG_TOOL_DATA=1")


async def demo_logging_configuration():
    print("\n🪵 Logging Configuration")

    print("🔧 Default Behavior:")
    print("   - Only warnings/errors are shown by default.")
    print("   - Other debug logs are suppressed.")

    print("\n🔊 Enabling Verbose Logging:")
    try:
        enable_verbose_stdout_logging()
        print("✅ Verbose logging enabled!")

        # Inspect the logger
        logger = logging.getLogger("openai.agents")
        print(f"🔍 Logger Level: {logging.getLevelName(logger.level)}")
        print(f"🧰 Logger has {len(logger.handlers)} handler(s)")

    except Exception as e:
        print(f"❌ Failed to enable verbose logging: {e}")

    print("\n🛠️ Custom Logging (if you want deeper control):")
    print(
        """
   import logging
   logger = logging.getLogger('openai.agents')
   logger.setLevel(logging.DEBUG)
   handler = logging.StreamHandler()
   logger.addHandler(handler)
    """
    )

    print("💡 There’s also a separate logger for tracing: 'openai.agents.tracing'")


@function_tool
def add_numbers(x: int, y: int) -> str:
    """Simple tool to add two numbers."""
    return f"{x} + {y} = {x + y}"


async def demo_configured_agent():
    print("\n🤖 Running Test Agent")

    agent = Agent(
        name="ConfigCheckerBot",
        instructions="""
        You are a friendly assistant that performs basic math using tools.
        Always keep the response clear and short.
        """,
        tools=[add_numbers],
        model=model
    )

    try:
        response = await Runner.run(
            agent, "Please add 40 and 2 using the tool.", max_turns=2
        )

        print("✅ Agent executed successfully!")
        print(f"📤 Final Output: {response.final_output}")

        print("\n📈 Run Metadata:")
        print(f"   - New Items: {len(response.new_items)}")
        print(f"   - Raw Responses: {len(response.raw_responses)}")
        print(f"   - Last Agent: {response.last_agent.name}")

    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        print("💡 Tip: This could be due to API key, rate limit, or network issues.")

async def demo_environment_variables():
    print("\n🌿 [Step 6] Environment Variables Overview")

    env_info = {
        "OPENAI_API_KEY": "Your OpenAI API key (required)",
        "OPENAI_API_BASE": "Custom API base URL (optional)",
        "OPENAI_AGENTS_DONT_LOG_MODEL_DATA": "Disables LLM input/output logging (optional)",
        "OPENAI_AGENTS_DONT_LOG_TOOL_DATA": "Disables tool input/output logging (optional)"
    }

    for var, desc in env_info.items():
        value = os.getenv(var)
        status = "✅ SET" if value else "❌ NOT SET"
        print(f"\n🔹 {var}: {status}")
        print(f"   📘 Description: {desc}")
        if value and var != "OPENAI_API_KEY":
            print(f"   📥 Value: {value}")

async def main():
    print("🚀 My OpenAI Agents SDK Setup Demo")

    # await demo_basic_configuration()
    # await demo_tracing_configuration()
    # await demo_logging_configuration()
    # await demo_configured_agent()
    await demo_environment_variables()


asyncio.run(main())
