# Standard Library Imports
import os
import json
from dataclasses import dataclass
from typing import Any, Optional, List, Dict
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel

# Third-party Imports
from agents import (
    Agent,
    Runner,
    AgentOutputSchema,
    set_tracing_disabled,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    AgentOutputSchemaBase,
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

# =====================================
# 1. Example: Non-strict Output Schema
# =====================================

class UserContextResult1(BaseModel):
    """Schema for user context result. Not strict-compatible due to dict usage."""
    enrolled: bool
    enrolled_courses: dict | None = None
    next: str | None = None
    message: str | None = None

input_data = (
    "I have fetched the user context. The user is enrolled in the course 'Fundamentals of Agentic AI' (AI-201) "
    "and is assigned to section 'UR-2D-2' (section code: UR-AI-201). The user's student ID is SAQIB2."
)

agent1 = Agent(
    name="json converter",
    instructions="convert to json text",
    output_type=AgentOutputSchema(UserContextResult1, strict_json_schema=False),
    model=model,
)

# =============================
# 2. OutputType Example Schema
# =============================

class OutputType(BaseModel):
    """A list of jokes, indexed by joke number (integer)."""
    jokes: dict[int, str]

# =====================================
# 3. Custom Output Schema Demonstration
# =====================================

class CustomOutputSchema(AgentOutputSchemaBase):
    """Custom output schema for demonstration purposes."""
    def is_plain_text(self) -> bool:
        return False
    def name(self) -> str:
        return "CustomOutputSchema"
    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "jokes": {"type": "object", "properties": {"joke": {"type": "string"}}}
            },
        }
    def is_strict_json_schema(self) -> bool:
        return False
    def validate_json(self, json_str: str) -> Any:
        json_obj = json.loads(json_str)
        # Return a list of jokes for demonstration
        return list(json_obj["jokes"].values())

# =====================================
# 4. Standardized Agent Runs & Debugging
# =====================================

input = "Tell me 3 short jokes."

print("\n[STEP] Running UserContextResult1 agent (non-strict)")
try:
    result = Runner.run_sync(agent1, input_data)
    print("  [RESULT] UserContextResult1 agent:", result.final_output)
except Exception as e:
    print("  [ERROR] UserContextResult1 agent:", e)

print("\n[STEP] Running OutputType agent (strict, should fail)")
try:
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        output_type=OutputType,
        model=model,
    )
    result = Runner.run_sync(agent, input)
    print("  [RESULT] OutputType agent (strict):", result.final_output)
    raise AssertionError("Should have raised an exception")
except Exception as e:
    print("  [EXPECTED ERROR] OutputType agent (strict):", e)

print("\n[STEP] Running OutputType agent (non-strict)")
try:
    agent.output_type = AgentOutputSchema(OutputType, strict_json_schema=False)
    result = Runner.run_sync(agent, input)
    print("  [RESULT] OutputType agent (non-strict):", result.final_output)
except Exception as e:
    print("  [ERROR] OutputType agent (non-strict):", e)

print("\n[STEP] Running CustomOutputSchema agent")
try:
    agent.output_type = CustomOutputSchema()
    result = Runner.run_sync(agent, input)
    print("  [RESULT] CustomOutputSchema agent:", result.final_output)
except Exception as e:
    print("  [ERROR] CustomOutputSchema agent:", e)

# =====================================
# 5. Project Idea: Tweet Summarizer Agent
# =====================================

class TweetSummarySchema(AgentOutputSchemaBase):
    """Output schema for tweet summarizer: expects a list of summaries."""
    def is_plain_text(self) -> bool:
        return False
    def name(self):
        return "TweetSummarySchema"
    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"summaries": {"type": "array", "items": {"type": "string"}}},
        }
    def is_strict_json_schema(self) -> bool:
        return False
    def validate_json(self, json_str: str) -> Any:
        try:
            json_obj = json.loads(json_str)
            return json_obj["summaries"]
        except Exception:
            lines = json_str.strip().split("\n")
            return [line.strip("- ").strip() for line in lines if line.strip()]

# Instantiate the tweet summarizer agent

tweet_agent = Agent(
    name="tweet-summarizer",
    instructions='Read the paragraph and summarize each of the 3 tweets. Return the summaries in a JSON list format like this: { "summaries": ["...", "...", "..."] }',
    output_type=TweetSummarySchema(),
    model=model,
)

tweet_paragraph = (
    "Elon Musk just tweeted that X.ai will open-source its latest language model this week! "
    "Meanwhile, OpenAI announced its new collaboration with Apple to integrate ChatGPT across iOS devices. "
    "In other news, DeepMind has shared a massive robotics dataset designed to improve reinforcement learning."
)

print("\n[STEP] Running Tweet Summarizer Agent")
try:
    result = Runner.run_sync(tweet_agent, tweet_paragraph)
    print("  [RESULT] Tweet Summarizer Agent:", result.final_output)
except Exception as e:
    print("  [ERROR] Tweet Summarizer Agent:", e)
