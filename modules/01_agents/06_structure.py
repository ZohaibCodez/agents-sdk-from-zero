"""
Comprehensive Guide: Structured Outputs in OpenAI Agents SDK

This file demonstrates 8 different approaches to structured outputs, from basic to advanced,
showing when to use each approach and the trade-offs involved.

Key Concepts:
1. Strict vs Non-Strict Schemas
2. Simple vs Complex Data Structures
3. Optional vs Required Fields
4. Nested Models vs Flat Structures
5. Performance vs Flexibility Trade-offs
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Optional, Union, Literal, Any
from enum import Enum
from datetime import datetime
from agents import (
    Agent,
    Runner,
    AgentOutputSchema,
    set_tracing_disabled,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
)
import asyncio
import os
from dotenv import load_dotenv, find_dotenv

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


# =============================================================================
# USE CASE 1: Basic Strict Schema (Recommended for Production)
# =============================================================================


class BasicAnimeInfo(BaseModel):
    """
     ✅ STRICT MODE COMPATIBLE
    - Extracts simple structured info about a book
    """

    title: str = ""
    episodes: int = 0
    is_completed: bool = False
    genre: str = ""

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# USE CASE 2: Enum-Based Strict Schema (Great for Classification)
# =============================================================================


class PowerLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CharacterRole(str, Enum):
    MAIN = "main"
    SUPPORT = "support"
    ANTAGONIST = "antagonist"


class CombatType(str, Enum):
    PHYSICAL = "physical"
    NINJUTSU = "ninjutsu"
    SPIRITUAL = "spiritual"


class AnimeCharacterClassification(BaseModel):
    name: str = ""
    role: CharacterRole = CharacterRole.MAIN
    power_level: PowerLevel = PowerLevel.MEDIUM
    combat_type: CombatType = CombatType.NINJUTSU
    affiliation: Literal["leaf", "akatsuki", "sand", "rogue"] = "leaf"

    model_config = ConfigDict(extra="forbid")


sample_inputs: dict[str, str] = {
    "anime_basic": "Naruto is a long-running shonen anime with 220 episodes in the original series. It is completed and primarily belongs to the action genre.",
    "character_classification": " Itachi Uchiha is an antagonist ninja from the Akatsuki. He uses powerful genjutsu and ninjutsu. His strength is on high-level.",
}

agents = {
    "anime_basic": Agent(
        name="AnimeInfoExtractor",
        instructions="Extract basic anime info into the specified format.",
        output_type=BasicAnimeInfo,
        model=model,
    ),
    "character_classifier": Agent(
        name="AnimeCharacterClassifier",
        instructions="Extract character name, role (main/support/antagonist), power level, combat type, and affiliation.",
        output_type=AnimeCharacterClassification,
        model=model,
    ),
}


async def run_comprehensive_tests():
    """
    Run all current structured output tests — anime edition.
    """

    test_cases = [
        ("anime_basic", "anime_basic", "Anime Metadata Extractor (Strict Schema)"),
        (
            "character_classifier",
            "character_classification",
            "Anime Character Classifier (Enum-Based)",
        ),
    ]

    print("=" * 80)
    print("🎓 STRUCTURED OUTPUT TESTING — ANIME EDITION")
    print("=" * 80)

    for i, (agent_key, input_key, description) in enumerate(test_cases, 1):
        print(f"\n{'=' * 20} USE CASE {i}: {description} {'=' * 20}")

        try:
            agent = agents[agent_key]
            input_data = sample_inputs[input_key]

            print(f"📝 Input: {input_data.strip()[:100]}...")
            print(f"🤖 Agent: {agent.name}")
            print(f"📋 Output Type: {agent.output_type}")

            result = await Runner.run(agent, input_data)

            print(f"✅ Success!")
            print(f"📊 Result: {result.final_output}")
            print(f"🔍 Type: {type(result.final_output)}")

            if hasattr(result.final_output, "__dict__"):
                print(f"📁 Structured Fields:")
                for field, value in result.final_output.__dict__.items():
                    if hasattr(value, "__dict__"):
                        print(f"   {field}: {type(value).__name__} -> {value}")
                    else:
                        print(f"   {field}: {value}")

        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"🔧 This demonstrates a limitation or mismatch in schema/data")

        print("-" * 80)


asyncio.run(run_comprehensive_tests())
