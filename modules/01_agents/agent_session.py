import asyncio
from openai import AsyncOpenAI
from agents import Agent, Runner, SQLiteSession, OpenAIChatCompletionsModel
from dotenv import find_dotenv, load_dotenv
import os


load_dotenv(find_dotenv())
# set_tracing_disabled(True)

# Configuration constants
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# Validate API key
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY environment variable is required but not found.")
    raise ValueError("GEMINI_API_KEY environment variable is required but not found.")

print(f"🚀 Initializing Gemini client with model: gemini-2.0-flash")

# Initialize external OpenAI client for Gemini
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Configure the chat completions model
model = OpenAIChatCompletionsModel(
    openai_client=external_client, model="gemini-2.0-flash"
)

print(f"✅ Model configured successfully\n")

agent = Agent(name="Assistant", instructions="Reply very concisely.", model=model)

# session = SQLiteSession("conversation_123")


# result = Runner.run_sync(
#     agent,
#     "Hi, My name is zohaib and i am a future data scientist and agentic ai engineer?",
#     session=session
# )
# print(result.final_output)

# result = Runner.run_sync(
#     agent,
#     "what do you know about me",
#     session=session
# )
# print(result.final_output)  # "California"

# async def main():
#     session = SQLiteSession("correction_example")
#     # Initial conversation
#     result = await Runner.run(
#         agent,
#         "What's 2 + 2?",
#         session=session
#     )
#     print(f"Agent: {result.final_output}")

#     # User wants to correct their question
#     assistant_item = await session.pop_item()  # Remove agent's response
#     user_item = await session.pop_item()  # Remove user's question

#     print(assistant_item)
#     print("="*10)
#     print(user_item)

#     # # Ask a corrected question
#     # result = await Runner.run(
#     #     agent,
#     #     "What's 2 + 3?",
#     #     session=session
#     # )
#     # print(f"Agent: {result.final_output}")


async def main():
    # Create an agent
    agent = Agent(name="Assistant", instructions="Reply very concisely.", model=model)

    # Create a session instance that will persist across runs
    session = SQLiteSession("conversation_123", "conversation_history.db")

    print("=== Sessions Example ===")
    print("The agent will remember previous messages automatically.\n")

    # First turn
    print("First turn:")
    print("User: What city is the Golden Gate Bridge in?")
    result = await Runner.run(
        agent, "What city is the Golden Gate Bridge in?", session=session
    )
    print(f"Assistant: {result.final_output}")
    print()

    # Second turn - the agent will remember the previous conversation
    print("Second turn:")
    print("User: What state is it in?")
    result = await Runner.run(agent, "What state is it in?", session=session)
    print(f"Assistant: {result.final_output}")
    print()

    # Third turn - continuing the conversation
    print("Third turn:")
    print("User: What's the population of that state?")
    result = await Runner.run(
        agent, "What's the population of that state?", session=session
    )
    print(f"Assistant: {result.final_output}")
    print()

    print("=== Conversation Complete ===")
    print("Notice how the agent remembered the context from previous turns!")
    print("Sessions automatically handles conversation history.")


asyncio.run(main())
