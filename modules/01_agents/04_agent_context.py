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
from dotenv import load_dotenv

load_dotenv()

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


def create_instruction(ctx: RunContextWrapper[str], agent: Agent[str]) -> str:
    print(f"[CONTEXT]:{ctx.context}")
    return f"You are a helpful assistant that provides concise answers. You are talking to {ctx.context}"


agent_basic: Agent[str] = Agent[str](
    name="Context Agent", instructions=create_instruction, model=model
)


async def main():
    name = "Zohaib"
    result: RunResult = await Runner.run(agent_basic, "Hi", context=name)
    print(result.final_output)


asyncio.run(main())
