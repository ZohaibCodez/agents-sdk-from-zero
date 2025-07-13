"""
modules/06_handoffs/05_handoff_input_filters.py

University Support Agent Handoff Demo
-------------------------------------
Demonstrates advanced agent handoffs with input filters for privacy, redaction, and summarization.

Features:
- Multiple handoff types (raw, redacted, summarized, privacy)
- Custom input filters for FERPA/GDPR compliance
- Professional terminal output with banners and emoji
- Robust error handling and clear test/demo cases

Environment Variables:
- GEMINI_API_KEY: API key for Gemini model

Author: Zohaib Khan
"""

# ============================
# Imports and Setup
# ============================
import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from agents.extensions.models.litellm_model import LitellmModel
from agents.extensions import handoff_filters
from agents import (
    Agent,
    ItemHelpers,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    enable_verbose_stdout_logging,
    function_tool,
    handoff,
    set_tracing_disabled,
    HandoffInputData,
)
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


# =========================
# Verbose Logging (LLM Debug Mode)
# =========================
# To see detailed message history, tool calls, and handoff context in the terminal output,
# uncomment the following line. This is very useful for debugging and educational purposes!
# enable_verbose_stdout_logging()

# ============================
# Environment and Model Setup
# ============================
load_dotenv(find_dotenv())
set_tracing_disabled(True)

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_NAME: str = "gemini-2.0-flash"

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY environment variable is required but not found.")
    raise ValueError("GEMINI_API_KEY environment variable is required but not found.")

print(f"\n==============================")
print(f"🚀 Initializing Gemini client with model: {GEMINI_MODEL_NAME}")
print(f"==============================\n")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL,
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client, model=GEMINI_MODEL_NAME
)

print(f"✅ Model configured successfully\n")


# ============================
# Tool Definitions
# ============================
@function_tool
def fetch_student_info(student_id: str) -> str:
    """Retrieve student academic profile.

    Args:
        student_id (str): The student's unique identifier.
    Returns:
        str: Academic profile summary.
    """
    return f"Student {student_id}: Enrolled in BSc Data Science, GPA 3.8, Semester 2"


@function_tool
def log_support_request(category: str, description: str) -> str:
    """Log a support ticket.

    Args:
        category (str): Ticket category.
        description (str): Ticket description.
    Returns:
        str: Confirmation message.
    """
    return f"Ticket logged under {category}: {description}"


# ============================
# Input Filter Functions
# ============================
def redact_academic_info(input_data: HandoffInputData) -> HandoffInputData:
    """Redact sensitive academic info (GPA, ID, email) from input history.

    Args:
        input_data (HandoffInputData): The handoff input data.
    Returns:
        HandoffInputData: Redacted input data.
    """
    import re

    def clean(text: str) -> str:
        # GPA patterns
        text = re.sub(
            r"\bGPA\b[:\s]*\d\.\d", "[GPA_REDACTED]", text, flags=re.IGNORECASE
        )
        # ID patterns like "Student ID: 12345678" or "ID 12345678"
        text = re.sub(
            r"\b(Student\s*ID|ID)\b[:\s]*\d{6,10}",
            "[ID_REDACTED]",
            text,
            flags=re.IGNORECASE,
        )
        # Emails
        text = re.sub(
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL_REDACTED]", text
        )
        return text

    cleaned_history = (
        clean(input_data.input_history)
        if isinstance(input_data.input_history, str)
        else input_data.input_history
    )

    return HandoffInputData(
        input_history=cleaned_history,
        pre_handoff_items=input_data.pre_handoff_items,
        new_items=input_data.new_items,
    )


def summarize_student_issue(input_data: HandoffInputData) -> HandoffInputData:
    """Summarize student escalation for Dean's Office handoff.

    Args:
        input_data (HandoffInputData): The handoff input data.
    Returns:
        HandoffInputData: Summarized input data.
    """
    summary = """
    STUDENT ESCALATION SUMMARY:
    - Reported repeated unresolved issues
    - Interacted with multiple departments
    - Now requesting direct action from Dean
    - Refer to structured logs for context
    """
    return HandoffInputData(
        input_history=summary.strip(),
        pre_handoff_items=input_data.pre_handoff_items,
        new_items=input_data.new_items,
    )


def remove_all_history(input_data: HandoffInputData) -> HandoffInputData:
    """Remove all conversation history for privacy officer handoff.

    Args:
        input_data (HandoffInputData): The handoff input data.
    Returns:
        HandoffInputData: Input data with history removed.
    """
    message = """
    PRIVACY NOTICE:
    - History removed per student request
    - Only new structured context is provided
    """
    return HandoffInputData(
        input_history=message.strip(),
        pre_handoff_items=(),
        new_items=input_data.new_items,
    )


# ============================
# Agent Definitions
# ============================
# 🔹 IT Support Agent
it_agent = Agent(
    name="IT Support",
    instructions="""You are the university IT support agent.
    Handle login issues, Wi-Fi access, and student portal problems.
    You receive filtered conversation history with tool logs removed for clarity.""",
    model=model,
)

# 🔹 Course Advisor Agent
course_advisor = Agent(
    name="Course Advisor",
    instructions="""You help students with course enrollments, drops, and schedule conflicts.
    Sensitive academic data (GPA, ID) may be redacted for compliance.""",
    model=model,
)

# 🔹 Privacy Officer Agent
privacy_agent = Agent(
    name="Privacy Officer",
    instructions="""You handle FERPA and student data privacy concerns.
    You receive no conversation history due to privacy policy.""",
    model=model,
)

# 🔹 Dean’s Office Agent
dean_agent = Agent(
    name="Dean's Office",
    instructions="""You review critical escalations and major complaints.
    You only receive summarized student issues for high-level action.""",
    model=model,
)

main_agent: Agent = Agent(
    name="Student Support Agent",
    instructions="""You are the main university support agent. 
    Respond to student queries and forward them to relevant departments 
    with appropriate filters applied based on sensitivity.""",
    tools=[fetch_student_info, log_support_request],
    model=model,
    handoffs=[
        # 🔹 Raw handoff to Course Advisor (no filters)
        course_advisor,
        # 🔹 Handoff to IT Support — filter out tool logs
        handoff(
            agent=it_agent,
            tool_name_override="contact_it_support",
            tool_description_override="Send to IT with tool logs removed",
            input_filter=handoff_filters.remove_all_tools,
        ),
        # 🔹 Handoff to Course Advisor — redacting sensitive info
        handoff(
            agent=course_advisor,
            tool_name_override="contact_course_advisor_redacted",
            tool_description_override="Send to Course Advisor with GPA and email redacted",
            input_filter=redact_academic_info,
        ),
        # 🔹 Handoff to Dean — summarized context
        handoff(
            agent=dean_agent,
            tool_name_override="escalate_to_dean",
            tool_description_override="Escalate to Dean's Office with summary only",
            input_filter=summarize_student_issue,
        ),
        # 🔹 Handoff to Privacy — no history at all
        handoff(
            agent=privacy_agent,
            tool_name_override="forward_to_privacy_officer",
            tool_description_override="Forward to Privacy Officer with no previous history",
            input_filter=remove_all_history,
        ),
    ],
)


# ============================
# Test/Demo Cases
# ============================
async def test_case_1():
    """Test raw handoff to Course Advisor."""
    print("\n" + "=" * 50)
    print("📚 Test 1: Raw Handoff to Course Advisor")
    print("=" * 50)
    try:
        input_text = "I want to swap my 'Statistics' class with 'Data Ethics'. Can you send this to a course advisor?"
        print(f"🔹 Input: {input_text}")
        result1 = await Runner.run(
            main_agent,
            input=input_text,
        )
        print(f"✅ Output: {result1.final_output}\n")
    except Exception as e:
        print(f"❌ Error in Test 1: {e}\n")


async def test_case_2():
    """Test IT handoff with tool logs removed."""
    print("\n" + "=" * 50)
    print("🧑‍💻 Test 2: IT Handoff (Tool Logs Removed)")
    print("=" * 50)
    try:
        input_text1 = (
            # "Check my student profile first. Then I need help logging into the portal."
            "Hi, I'm Student ID: 12345678 and my GPA is 3.9.\n"
            "I want to drop 'Linear Algebra' and enroll in 'AI Ethics'.\n"
            "Please notify the course advisor."
        )
        print(f"🔹 Input: {input_text1}")
        result2a = await Runner.run(
            main_agent,
            input=input_text1,
        )
        input_text2 = result2a.to_input_list() + [
            {"role": "user", "content": "Please transfer me to IT support."}
        ]
        print(f"🔹 Input: {input_text2}")
        result2b = await Runner.run(main_agent, input=input_text2)
        print(f"✅ Output: {result2b.final_output}\n")
    except Exception as e:
        print(f"❌ Error in Test 2: {e}\n")


async def test_case_3():
    """Test redacted handoff to Course Advisor."""
    print("\n" + "=" * 50)
    print("📝 Test 3: Redacted Handoff to Course Advisor")
    print("=" * 50)
    try:
        input_text = """Hi, I'm Student ID: 12345678 and my GPA is 3.9.\n"
                      "I want to drop 'Linear Algebra' and enroll in 'AI Ethics'.\n"
                      "Please notify the course advisor."""
        print(f"🔹 Input: {input_text}")
        result3 = await Runner.run(
            main_agent,
            input=input_text,
        )
        print(f"✅ Output: {result3.final_output}\n")
    except Exception as e:
        print(f"❌ Error in Test 3: {e}\n")


async def test_case_4():
    """Test escalation to Dean's Office."""
    print("\n" + "=" * 50)
    print("🏛️ Test 4: Escalation to Dean")
    print("=" * 50)
    try:
        input_text = """I've had multiple unresolved issues with scheduling, portal bugs, and no one replies.\n"
                      "I need this escalated to the Dean's Office immediately."""
        print(f"🔹 Input: {input_text}")
        result4 = await Runner.run(main_agent, input=input_text)
        print(f"✅ Output: {result4.final_output}\n")
    except Exception as e:
        print(f"❌ Error in Test 4: {e}\n")


async def test_case_5():
    """Test privacy officer handoff (no history)."""
    print("\n" + "=" * 50)
    print("🔒 Test 5: Privacy Officer (No History)")
    print("=" * 50)
    try:
        input_text = """I want to speak about how my student data is stored and whether it's FERPA compliant.\n"
                      "I request this conversation not be shared. Please forward to the privacy officer."""
        print(f"🔹 Input: {input_text}")
        result5 = await Runner.run(main_agent, input=input_text)
        print(f"✅ Output: {result5.final_output}\n")
    except Exception as e:
        print(f"❌ Error in Test 5: {e}\n")


# ============================
# Main Entrypoint
# ============================
async def main():
    """Main entrypoint for University Support Input Filters Demo."""
    print("\n" + "#" * 60)
    print("🎓 University Support Input Filters Demo")
    print("#" * 60 + "\n")
    # Uncomment the tests you want to run:
    # await test_case_1()
    await test_case_2()
    # await test_case_3()
    # await test_case_4()
    # await test_case_5()
    print("\n" + "#" * 60)
    print("🏁 Demo complete.")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
