"""
08_runner_chat.py

Demonstrates managing multi-turn conversations (chat threads) with Runner.

Features demonstrated:
- Running an initial turn with an agent.
- Using RunResultBase.to_input_list() to get conversation history.
- Appending new user messages to the history for follow-up turns.
- Running subsequent turns with the updated history.
- Using RunConfig.group_id to link traces for a conversation thread.

Based on: https://openai.github.io/openai-agents-python/running_agents/#conversationschat-threads

Environment variables:
    - OPENROUTER_API_KEY: Required for OpenRouter API access.

Author: Zohaib Khan
"""

import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

# =========================
# Environment & Model Setup
# =========================

load_dotenv(find_dotenv())
set_tracing_disabled(True)

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    print("OPENROUTER_API_KEY not found.")

external_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client, model="deepseek/deepseek-chat-v3-0324:free"
)

# =========================
# Conversational Agent Setup
# =========================

conversational_agent = Agent(
    name="ConversationalAgent",
    instructions="You are a helpful and concise conversational AI. Remember previous parts of the conversation.",
    model=model,
)

# =========================
# Main Demo Function
# =========================

async def main():
    """
    Demonstrates a multi-turn conversation with an agent, using conversation history.
    - Runs an initial turn.
    - Uses to_input_list() to get conversation history.
    - Appends a new user message and runs a follow-up turn.
    - Prints outputs and conversation history for clarity.
    """
    print("\n--- Running 08_runner_chat.py ---\n")
    agent = Agent(name="Assistant", instructions="Reply very concisely.", model=model)

    # First turn
    print("="*60)
    print("[Turn 1] User: Hello! My name is zohaib. I am a Data science student currently undergraduate and also doing some learning stuff in currently emerging tech named Agentic AI.")
    result = await Runner.run(
        agent,
        "Hello! My name is zohaib. I am a Data science student currently undergraduate and also doing some learning stuff in currently emerging tech named Agentic AI.",
    )
    print("[Assistant]:", result.final_output)
    print("\n[Conversation History after Turn 1]:")
    print(result.to_input_list())

    # Second turn
    print("\n" + "="*60)
    print("[Turn 2] User: What is my name? Do you remember me ?")
    new_input = result.to_input_list() + [
        {"role": "user", "content": "What is my name? Do you remember me ?"}
    ]
    result = await Runner.run(agent, new_input)
    print("[Assistant]:", result.final_output)
    print("\n[Conversation History after Turn 2]:")
    print(result.to_input_list())

    print("\n--- Finished 08_runner_chat.py ---\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        print(f"Error type: {type(e).__name__}")
