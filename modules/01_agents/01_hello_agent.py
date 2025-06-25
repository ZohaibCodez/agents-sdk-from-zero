from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    RunResult,
    RunContextWrapper,
)
import asyncio
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

set_tracing_disabled(True)
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    print("GEMINI_API_KEY not found.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client, model="gemini-2.0-flash"
)


async def main():
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        model=model,
    )

    result = await Runner.run(agent, "HI!")
    print(result.final_output)


asyncio.run(main())
