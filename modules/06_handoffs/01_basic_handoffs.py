"""
modules/06_handoffs/01_basic_handoffs.py

Demonstrates advanced agent handoffs in a university support scenario using OpenAI Agents SDK.
Features:
- Multi-agent handoff (triage to Enrollment, Finance, Tech)
- Streaming, error handling, and robust output formatting
- Professional, educational code structure

Environment Variables:
- GEMINI_API_KEY: API key for Gemini model

Author: Zohaib Khan
Reference: https://github.com/openai/openai-agents-sdk
"""
# =========================
# Imports & Setup
# =========================
import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from agents.extensions.models.litellm_model import LitellmModel
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

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

print(f"\n==============================")
print(f"🚀 Initializing Gemini client with model: {GEMINI_MODEL_NAME}")
print(f"==============================\n")

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

# =========================
# Main Demo Function
# =========================
async def main():
    """
    Demonstrates agent handoffs in a university support system.
    Runs a triage agent that hands off to Enrollment, Finance, or Tech agents based on student input.
    Prints clearly formatted results for each test case.
    """
    print("\n==============================")
    print("🎓 University Support System Demo")
    print("==============================\n")
    # --- Agent Definitions ---
    enrollment_agent = Agent(
        name="Enrollment Advisor",
        instructions="""You are responsible for:
        - New student admissions
        - Course enrollment
        - Changing or dropping classes
        \nAlways provide clear steps and university policy when helping students with enrollment tasks.""",
        model=model,
        handoff_description="Handles course registration, admissions, and class changes.",
        model_settings = ModelSettings(max_tokens=150),
    )

    finance_agent = Agent(
        name="Finance Officer",
        instructions="""You help students with:
        - Tuition payment problems
        - Scholarship or financial aid queries
        - Billing errors or fee explanations
        \nBe precise and clear when discussing financial matters.""",
        model=model,
        handoff_description="Responsible for tuition issues, scholarships, and student fee problems.",
        model_settings = ModelSettings(max_tokens=150)
    )

    tech_agent = Agent(
        name="Tech Helpdesk",
        instructions="""You assist students with:
        - Login problems
        - University email setup
        - LMS (Learning Management System) access or bugs
        \nProvide clear troubleshooting steps.""",
        model=model,
        handoff_description="Provides tech support for LMS, email, login issues, and student portal access.",
        model_settings = ModelSettings(max_tokens=150)
    )

    triage_agent = Agent(
        name="Student Support Bot",
        instructions="""You are a university triage bot. Your job is to:
        1. Understand the student's issue
        2. Transfer them to the right department
        3. Set expectations for what kind of help they'll receive
        \nSpecialists available:
        - Enrollment Advisor: Admissions, courses, schedule changes
        - Finance Officer: Tuition, billing, scholarships
        - Tech Helpdesk: Login issues, email problems, LMS bugs
        \nAlways explain why you're transferring them.""",
        handoffs=[enrollment_agent, finance_agent, tech_agent],
        model=model,
        model_settings = ModelSettings(max_tokens=150)
    )

    test_inputs = {
        "Enrollment": "I need help registering for next semester's courses.",
        "Finance": "I was charged twice for my tuition. How do I fix this?",
        "Tech": "My LMS isn't loading and I can't submit my assignment.",
        "Ambiguous": "Something's wrong with my student account.",
        "Cross-domain": "I need to change my course and update my payment method.",
        "Irrelevant": "Can you tell me who won the football match yesterday?",
    }

    for test_name, message in test_inputs.items():
        print(f"\n==============================")
        print(f"🧪 Test Case: {test_name}")
        print(f"------------------------------")
        print(f"📥 Student Input: {message}")
        try:
            result = await Runner.run(triage_agent, input=message)
            print(f"\n📤 Student Bot Output:")
            print(f"{result.final_output}")
            print(f"✅ Success: Response generated.")
        except Exception as e:
            print(f"❌ Error during agent run: {e}")
        print(f"==============================")

    print("\n==============================")
    print("🤝 Available Handoff Agents")
    print("==============================")
    for agent in getattr(triage_agent, "handoffs", []):
        name = getattr(agent, "name", "<Unknown>")
        desc = getattr(agent, "handoff_description", "<No description>")
        print(f"\n🔗 Handoff Agent Name: {name}")
        print(f"   Description: {desc}")
    print("\n🎉 Demo complete!\n")

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    asyncio.run(main())
