"""
modules/06_handoffs/02_handoffs_with_callbacks.py

Demonstrates advanced agent handoffs with custom callbacks in a university student services scenario using OpenAI Agents SDK.
Features:
- Multi-agent handoff with custom tool names and callback functions
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
    RunContextWrapper,
    Runner,
    handoff,
    set_tracing_disabled,
)
from openai import AsyncOpenAI
from pydantic import BaseModel

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
# Data Models
# =========================
class ITIssue(BaseModel):
    """Represents an IT issue for handoff to IT support agent."""
    system: str
    description: str

# =========================
# Callback Functions
# =========================
def on_academic_handoff(ctx: RunContextWrapper[None]):
    """Callback for academic handoff."""
    print("\n📚 Academic Handoff Triggered")
    print(f"Agent Context: {ctx}")


def on_it_handoff(ctx: RunContextWrapper[None], input_data: ITIssue):
    """Callback for IT handoff."""
    print("\n💻 IT Handoff Triggered")
    print(f"Agent Context: {ctx}")
    print(f"Reported IT Issue: {input_data}")


def on_financial_handoff(ctx: RunContextWrapper[None]):
    """Callback for financial handoff."""
    print("\n💸 Financial Handoff Triggered")
    print(f"Agent Context: {ctx}")


def on_affairs_handoff(ctx: RunContextWrapper[None]):
    """Callback for student affairs handoff."""
    print("\n⚠️ Student Affairs Handoff Triggered")
    print(f"Agent Context: {ctx}")

# =========================
# Agent Definitions
# =========================
academic_agent = Agent(
    name="Academic Advisor",
    instructions="Assist with course plans, prerequisites, and graduation queries.",
    model=model,
)

it_agent = Agent(
    name="IT Support",
    instructions="Help with portal access, system bugs, and technical issues.",
    model=model,
)

financial_agent = Agent(
    name="Financial Aid Officer",
    instructions="Handle scholarships, fee disputes, and financial aid applications.",
    model=model,
)

affairs_agent = Agent(
    name="Student Affairs",
    instructions="Handle complaints, wellbeing issues, and sensitive matters requiring escalation.",
    model=model,
)

main_agent = Agent(
    name="Student Services Assistant",
    instructions="""
You're a university student assistant bot. Route students to the right department:

- Use 'ask_academic_advisor' for course or graduation questions
- Use 'report_it_problem' for technical/system issues
- Use 'ask_financial_office' for fee or scholarship queries
- Use 'contact_student_affairs' for complaints or sensitive issues
""",
    model=model,
    handoffs=[
        handoff(
            agent=academic_agent,
            tool_name_override="ask_academic_advisor",
            tool_description_override="Connect to academic advisor for course, prerequisite, or credit questions.",
            on_handoff=on_academic_handoff,
        ),
        handoff(
            agent=it_agent,
            tool_name_override="report_it_problem",
            tool_description_override="Report system bugs, login failures, or tech problems.",
            on_handoff=on_it_handoff,
            input_type=ITIssue,
        ),
        handoff(
            agent=financial_agent,
            tool_name_override="ask_financial_office",
            tool_description_override="Ask about scholarships, financial aid, or fee issues.",
            on_handoff=on_financial_handoff,
        ),
        handoff(
            agent=affairs_agent,
            tool_name_override="contact_student_affairs",
            tool_description_override="Escalate to student affairs for complaints or personal concerns.",
            on_handoff=on_affairs_handoff,
        ),
    ],
)

# =========================
# Main Demo Function
# =========================
async def main():
    """
    Demonstrates agent handoffs with custom callbacks in a university student services system.
    Runs the main agent for several test cases and prints clearly formatted results.
    """
    print("\n==============================")
    print("🎓 Student Services Handoff System Demo")
    print("==============================\n")

    test_cases = [
        ("IT Issue", "I can't log into the portal. It says my credentials are wrong."),
        ("Academic Help", "Do I need to complete Data Structures before taking AI?"),
        ("Financial Query", "Can I apply for a scholarship if my GPA is 3.5?"),
        ("Escalation to Student Affairs", "I filed a harassment complaint last month and haven't received a response."),
    ]

    for idx, (test_name, message) in enumerate(test_cases, 1):
        print(f"\n==============================")
        print(f"🧪 Test Case {idx}: {test_name}")
        print(f"------------------------------")
        print(f"📥 Student Input: {message}")
        try:
            result = await Runner.run(main_agent, input=message)
            print(f"\n📤 Student Bot Output:")
            print(f"{result.final_output}")
            print(f"✅ Success: Response generated.")
        except Exception as e:
            print(f"❌ Error during agent run: {e}")
        print(f"==============================")

    print("\n🎉 Demo complete!\n")

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    asyncio.run(main())
