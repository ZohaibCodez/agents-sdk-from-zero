"""
Basic Function Tools Demo - OpenAI Agents SDK

This module demonstrates the fundamental usage of function tools with the OpenAI Agents SDK.
It showcases:
- Creating simple function tools with @function_tool decorator
- Using Pydantic models for structured tool parameters
- Tool introspection and metadata display
- Context management with RunContextWrapper
- Error handling and graceful failure

The demo uses a mock university database to demonstrate tool functionality.

Environment variables:
    - GEMINI_API_KEY: Required for Gemini model access.

Author: Zohaib Khan
"""

from openai import AsyncOpenAI
from agents import (
    Runner,
    Agent,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    RunResult,
    function_tool,
    RunContextWrapper,
    FunctionTool,
)
from dotenv import load_dotenv, find_dotenv
import os
from pydantic import BaseModel
from pprint import pprint

# =========================
# Environment & Model Setup
# =========================

load_dotenv(find_dotenv())
set_tracing_disabled(True)

# Configuration constants
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_NAME: str = "gemini-2.0-flash"

# Validate API key
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY environment variable is required but not found.")
    raise ValueError("GEMINI_API_KEY environment variable is required but not found.")

print(f"🚀 Initializing Gemini client with model: {GEMINI_MODEL_NAME}")

# Initialize external OpenAI client for Gemini
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL,
)

# Configure the chat completions model
model = OpenAIChatCompletionsModel(
    openai_client=external_client, model=GEMINI_MODEL_NAME
)

print(f"✅ Model configured successfully\n")


class UniversityQuery(BaseModel):
    """Pydantic model for university department queries."""

    university: str
    department: str


@function_tool
def get_university_info(name: str) -> str:
    """
    Retrieve information about a university.

    Args:
        name: The name or abbreviation of the university

    Returns:
        str: Information about the university or error message
    """
    universities = {
        "PUCIT": "Punjab University College of Information Technology, Lahore.",
        "FAST": "Foundation for Advancement of Science and Technology, Lahore.",
    }
    return universities.get(name, "No data found :(")


@function_tool
def get_department_info(query: UniversityQuery) -> str:
    """
    Retrieve information about a specific department at a university.

    Args:
        query: UniversityQuery object containing university and department names

    Returns:
        str: Information about the department or error message
    """
    db = {
        ("PUCIT", "CS"): "PUCIT's CS department offers BS, MS, and PhD in CS.",
        ("PUCIT", "DS"): "PUCIT's DS department specializes in data analytics and ML.",
        (
            "FAST",
            "CS",
        ): "FAST's CS department is known for its strong programming curriculum.",
    }
    return db.get((query.university, query.department), "No data found :(")


@function_tool
def read_syllabus(ctx: RunContextWrapper, filename: str) -> str:
    """
    Simulate reading a syllabus file and add the result to the context.

    Args:
        ctx: RunContextWrapper for managing execution context
        filename: Name of the syllabus file to read

    Returns:
        str: Syllabus content or error message
    """
    # Fake file database
    fake_files = {
        "PUCIT_DS.txt": "1. Intro to DS\n2. ML & Deep Learning\n3. Capstone Project",
        "FAST_CS.txt": "1. Algorithms\n2. Software Engineering\n3. Final Year Project",
    }

    content = fake_files.get(filename, "❌ File not found.")

    return f"📄 {filename}:\n{content}"


# =========================
# Agent Configuration
# =========================

agent: Agent = Agent(
    name="University Assistant",
    instructions="""
    You are a helpful University Assistant.
    Use `get_department_info` when the user asks about a department or its offerings.
    Use `get_university_info` when the user asks about a university in general.
    Use abbreviations like 'PUCIT' or 'FAST' as full names.
    - Assume 'PUCIT_DS.txt' and 'FAST_CS.txt' are valid files.
    """,
    tools=[get_university_info, get_department_info, read_syllabus],
    model=model,
)

# =========================
# Execute Agent Run
# =========================

print("🎓 Executing University Assistant Agent...")
try:
    result = Runner.run_sync(agent, "Can you show me the syllabus of PUCIT DS?")

    # ============================
    # 📋 Display Final Output
    # ============================
    print("\n🎓 Final Output:")
    print(result.final_output)

    # ============================
    # 🧾 Display Context Items (new_items)
    # ============================
    print("\n🧾 Context Items (Added by Tools):")
    for item in result.new_items:
        pprint(item)

    # ============================
    # 🔍 Tool Metadata Introspection
    # ============================
    for tool in agent.tools:
        print(f"🔧 Tool Name: {tool.name}")
        print(f"📘 Description: {tool.description}")
        print("📐 Params Schema:")
        print(tool.params_json_schema)
        print("-" * 40)

except Exception as e:
    print(f"❌ Error during agent execution: {e}")
    raise
