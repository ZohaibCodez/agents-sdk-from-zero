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


# =============================================================================
# USE CASE 3: Nested Strict Schema (Complex but Strict-Compatible)
# =============================================================================


class AffiliationInfo(BaseModel):
    village: str = ""
    organization: str = ""
    rank: Literal["genin", "chunin", "jonin", "kage", "rogue"] = "genin"

    model_config = ConfigDict(extra="forbid")


class Abilities(BaseModel):
    main_skill: str = ""
    type: Literal["ninjutsu", "taijutsu", "genjutsu", "senjutsu"] = "ninjutsu"
    mastery_level: int = 1

    model_config = ConfigDict(extra="forbid")


class AnimeCharacterProfile(BaseModel):
    basic_info: BasicAnimeInfo = Field(default_factory=BasicAnimeInfo)
    affiliation: AffiliationInfo = Field(default_factory=AffiliationInfo)
    abilities: Abilities = Field(default_factory=Abilities)
    debut_episode: str = ""

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# USE CASE 4: List-Based Strict Schema (Collections in Strict Mode)
# =============================================================================


class NinjaSkill(BaseModel):
    name: str = ""
    type: Literal["ninjutsu", "genjutsu", "taijutsu"] = "ninjutsu"
    mastery_level: int = 1

    model_config = ConfigDict(extra="forbid")


class NinjaSkillLog(BaseModel):
    ninja_name: str = ""
    village: str = ""
    skills: List[NinjaSkill] = Field(default_factory=list)
    total_skills: int = 0

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# USE CASE 5: Flexible Non-Strict Schema (Maximum Flexibility)
# =============================================================================


class BattleRecap(BaseModel):
    battle_name: str
    duration_minutes: Optional[int] = None
    outcome: Union[str, None] = None
    participants: Union[list[str], dict[str, str], None] = None
    highlights: Optional[list[str]] = None
    extra_notes: Optional[dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")  # non-strict


# =============================================================================
# USE CASE 6: Validated Strict Schema (Business Logic Validation)
# =============================================================================


class AnimeMission(BaseModel):
    mission_id: str = ""
    ninja_email: str = ""
    mission_rank: Literal["D", "C", "B", "A", "S"] = "D"
    reward: float = 0.0
    completed: bool = False

    model_config = ConfigDict(extra="forbid")

    @field_validator("ninja_email")
    @classmethod
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email address")
        return v

    @field_validator("reward")
    @classmethod
    def validate_reward(cls, v):
        if v < 0:
            raise ValueError("Reward must be non-negative")
        return v


# =============================================================================
# USE CASE 7: Multi-Level Nested Schema (Enterprise-Grade Structure)
# =============================================================================


class MissionHeader(BaseModel):
    mission_id: str = ""
    village: str = ""
    rank: Literal["D", "C", "B", "A", "S"] = "C"
    leader_email: str = ""

    model_config = ConfigDict(extra="forbid")


class NinjaMember(BaseModel):
    name: str = ""
    role: Literal["genin", "chuunin", "jounin", "sensei"] = "genin"
    specialty: str = ""

    model_config = ConfigDict(extra="forbid")


class TeamInfo(BaseModel):
    team_name: str = ""
    members: List[NinjaMember] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class TeamMission(BaseModel):
    mission: MissionHeader = Field(default_factory=MissionHeader)
    team: TeamInfo = Field(default_factory=TeamInfo)
    objective: str = ""
    completed: bool = False

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# USE CASE 8: Dynamic Schema Selection (Runtime Schema Choice)
# =============================================================================


class SimpleAnswer(BaseModel):
    message: str = "Acknowledged"
    success: bool = True

    model_config = ConfigDict(extra="forbid")


class DetailedAnswer(BaseModel):
    message: str = "Character details attached"
    success: bool = True
    character: AnimeCharacterProfile = Field(default_factory=AnimeCharacterProfile)
    context: BasicAnimeInfo = Field(default_factory=BasicAnimeInfo)

    model_config = ConfigDict(extra="forbid")


sample_inputs: dict[str, str] = {
    "anime_basic": "Naruto is a long-running shonen anime with 220 episodes in the original series. It is completed and primarily belongs to the action genre.",
    "character_classification": " Itachi Uchiha is an antagonist ninja from the Akatsuki. He uses powerful genjutsu and ninjutsu. His strength is on high-level.",
    "nested_character_profile": """
    Character Profile:
    Name: Kakashi Hatake, Age: 29, Power: High
    Affiliation: Village - Leaf, Organization - Anbu, Rank - Jonin
    Abilities: Lightning Blade (Ninjutsu), Mastery: 10
    Debuted in Episode 3
    """,
    "ninja_skills": """
    Ninja: Rock Lee from Leaf Village
    Skills:
    - Leaf Hurricane (Taijutsu), Mastery: 8
    - Strong Fist (Taijutsu), Mastery: 10
    - Front Lotus (Taijutsu), Mastery: 7
    Total: 3 Skills
    """,
    "battle_recap": """
    Battle: Naruto vs Pain
    Duration: ~45 minutes
    Outcome: Naruto wins after using Sage Mode and Rasenshuriken.
    Participants: Naruto (Hero), Pain (Antagonist)
    Highlights: Toad summons, Village destruction, Talk-no-jutsu
    Notes: This battle changed Naruto's reputation in the village.
    """,
    "validated_mission": "Mission ID: MIS-109, Email: kakashi@leaf.com, Rank: A, Reward: 15000, Completed: Yes",
    "team_mission": """
    Mission ID: MIS-2005
    Village: Leaf
    Rank: A
    Leader: kakashi@leaf.com
    Team: Team 7
    Members:
    - Naruto Uzumaki (Genin) - Shadow Clones
    - Sasuke Uchiha (Genin) - Fire Style
    - Sakura Haruno (Genin) - Medical Ninjutsu
    - Kakashi Hatake (Sensei) - Sharingan
    
    Objective: Escort the feudal lord to Hidden Sand.
    Completed: Yes""",
    "dynamic_response": "Tell me everything about Naruto Uzumaki including his age, contact, address, and registration date.",
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
    "nested_character_profile": Agent(
        name="CharacterProfileExtractor",
        instructions="Extract character profile including basic info, affiliation, abilities, and debut episode.",
        output_type=AnimeCharacterProfile,
        model=model,
    ),
    "ninja_skills": Agent(
        name="NinjaSkillAnalyzer",
        instructions="Extract the ninja's name, village, all known skills, and total number of skills.",
        output_type=NinjaSkillLog,
        model=model,
    ),
    "battle_recap": Agent(
        name="BattleRecapAnalyzer",
        instructions="Extract battle name, duration, outcome, participants, and highlights.",
        output_type=AgentOutputSchema(BattleRecap, strict_json_schema=False),
        model=model,
    ),
    "validated_mission": Agent(
        name="MissionValidator",
        instructions="Extract and validate anime mission info including ID, ninja email, rank, reward, and status.",
        output_type=AnimeMission,
        model=model,
    ),
    "team_mission": Agent(
        name="TeamMissionLogger",
        instructions="Extract mission header, team info, members, and objective details.",
        output_type=TeamMission,
        model=model,
    ),
    "anime_simple_responder": Agent(
        name="AnimeSimpleResponder",
        instructions="Give a simple confirmation response to anime-related queries.",
        output_type=SimpleAnswer,
        model=model,
    ),
    "anime_detailed_responder": Agent(
        name="AnimeDetailedResponder",
        instructions="Provide full character profile including personal info and context.",
        output_type=DetailedAnswer,
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
        (
            "nested_character_profile",
            "nested_character_profile",
            "Nested Anime Character Profile (Strict-Compatible)",
        ),
        ("ninja_skills", "ninja_skills", "Ninja Skill Log (List-Based Strict Schema)"),
        ("battle_recap", "battle_recap", "Flexible Battle Recap (Non-Strict Schema)"),
        (
            "validated_mission",
            "validated_mission",
            "Anime Mission Validator (Strict + Validation)",
        ),
        (
            "team_mission",
            "team_mission",
            "Team Mission Log (Multi-Level Nested Schema)",
        ),
        (
            "anime_simple_responder",
            "anime_basic",
            "Anime Simple Response (Dynamic Schema - Simple)",
        ),
        (
            "anime_detailed_responder",
            "dynamic_response",
            "Anime Detailed Response (Dynamic Schema - Detailed)",
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
