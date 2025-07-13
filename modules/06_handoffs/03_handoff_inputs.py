"""
modules/06_handoffs/03_handoff_inputs.py

University Student Support Assistant — Structured Handoffs Demo

This module demonstrates advanced usage of the OpenAI Agents SDK for structured agent handoffs in a university support scenario. It features:
- Multiple specialized agents (academic, tech, admissions, finance)
- Structured handoff with Pydantic models for data validation
- Callback functions for each handoff, with clear, visually distinct output
- Robust error handling and safe attribute access
- Professional, educational, and visually appealing terminal output
- Test/demo cases for each handoff scenario

Environment Variables:
- GEMINI_API_KEY: API key for Gemini model (required)

Author: Zohaib Khan
References:
- https://github.com/openai/openai-agents-sdk
"""

# ============================
# Imports and Setup
# ============================
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
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

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
# Data Models
# ============================
class UrgencyLevel(str, Enum):
    """Urgency levels for student issues."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AcademicConcernData(BaseModel):
    """Structured data for academic concerns."""

    issue: str = Field(..., description="Description of the academic issue")
    subject: str = Field(..., description="Subject or course name")
    deadline_impact: bool = Field(..., description="Is a deadline affected?")
    previous_help_attempts: list[str] = Field(
        default_factory=list, description="Steps already taken to resolve this"
    )


class PortalIssueData(BaseModel):
    """Structured data for LMS/portal technical issues."""

    error_message: str = Field(..., description="Exact error message seen")
    device_used: Optional[str] = Field(
        None, description="Device used (e.g., laptop, phone)"
    )
    browser: Optional[str] = Field(
        None, description="Browser or app where issue occurred"
    )
    urgency_level: UrgencyLevel = Field(
        default=UrgencyLevel.MEDIUM, description="Urgency of the issue"
    )
    actions_tried: list[str] = Field(
        default_factory=list, description="Troubleshooting already done"
    )


class AdmissionQueryData(BaseModel):
    """Structured data for admissions queries."""

    program_interest: str = Field(..., description="Program student is interested in")
    documents_submitted: Optional[List[str]] = Field(
        None, description="Documents already submitted"
    )
    application_status: Optional[str] = Field(
        None, description="Current known application status"
    )
    specific_questions: list[str] = Field(
        default_factory=list, description="Specific questions the student has"
    )


class FeeDisputeData(BaseModel):
    """Structured data for fee disputes."""

    fee_component: str = Field(
        ..., description="Component being disputed (e.g., exam fee)"
    )
    invoice_id: Optional[str] = Field(None, description="Invoice ID if available")
    amount_disputed: Optional[float] = Field(
        None, description="Exact amount in dispute"
    )
    reason: str = Field(..., description="Why the student is disputing the fee")
    payment_method: Optional[str] = Field(
        None, description="Method used for payment (if relevant)"
    )


# ============================
# Callback Functions for Handoffs
# ============================
async def on_academic_concern_with_data(
    ctx: RunContextWrapper[None], input_data: AcademicConcernData
):
    """
    Callback for academic concern handoff.
    Args:
        ctx: Run context wrapper (unused).
        input_data: AcademicConcernData instance.
    Returns: None
    """
    print("\n📘 Academic Concern (Callback):")
    print(f"   Issue: {input_data.issue}")
    print(f"   Subject: {input_data.subject}")
    print(f"   Deadline Impact: {'Yes' if input_data.deadline_impact else 'No'}")
    print(
        f"   Previous Attempts: {', '.join(input_data.previous_help_attempts) if input_data.previous_help_attempts else 'None'}"
    )
    # Imagine this updates a student issue tracking system


async def on_portal_issue_with_data(
    ctx: RunContextWrapper[None], input_data: PortalIssueData
):
    """
    Callback for LMS/portal technical issue handoff.
    Args:
        ctx: Run context wrapper (unused).
        input_data: PortalIssueData instance.
    Returns: None
    """
    print("\n🖥️ LMS/Portal Issue (Callback):")
    print(f"   Error Message: {input_data.error_message}")
    print(f"   Device: {input_data.device_used or 'Not specified'}")
    print(f"   Browser/App: {input_data.browser or 'Not specified'}")
    print(f"   Urgency: {input_data.urgency_level}")
    print(
        f"   Actions Tried: {', '.join(input_data.actions_tried) if input_data.actions_tried else 'None'}"
    )
    # Could log to a technical helpdesk ticketing tool


async def on_admission_query_with_data(
    ctx: RunContextWrapper[None], input_data: AdmissionQueryData
):
    """
    Callback for admissions query handoff.
    Args:
        ctx: Run context wrapper (unused).
        input_data: AdmissionQueryData instance.
    Returns: None
    """
    print("\n📄 Admission Query (Callback):")
    print(f"   Program: {input_data.program_interest}")
    print(
        f"   Submitted Docs: {', '.join(input_data.documents_submitted) if input_data.documents_submitted else 'None'}"
    )
    print(f"   Current Status: {input_data.application_status or 'Unknown'}")
    print(
        f"   Questions: {', '.join(input_data.specific_questions) if input_data.specific_questions else 'None'}"
    )
    # Could be used to prefill a response template for the counselor


async def on_fee_dispute_with_data(
    ctx: RunContextWrapper[None], input_data: FeeDisputeData
):
    """
    Callback for fee dispute handoff.
    Args:
        ctx: Run context wrapper (unused).
        input_data: FeeDisputeData instance.
    Returns: None
    """
    print("\n💳 Fee Dispute (Callback):")
    print(f"   Component: {input_data.fee_component}")
    print(f"   Invoice ID: {input_data.invoice_id or 'Not provided'}")
    print(f"   Amount: {input_data.amount_disputed or 'Not specified'}")
    print(f"   Reason: {input_data.reason}")
    print(f"   Payment Method: {input_data.payment_method or 'Unknown'}")
    # Could route info to a financial services team or billing portal


# ============================
# Agent Definitions
# ============================
academic_agent = Agent(
    name="Academic Advisor",
    instructions="You are an academic advisor. Help students with course-related concerns, missed deadlines, and academic struggles.",
    model=model,
    model_settings=ModelSettings(max_tokens=150),
)

tech_agent = Agent(
    name="Tech Support",
    instructions="You are a university tech support specialist. Resolve LMS, portal, and system access issues with technical accuracy.",
    model=model,
    model_settings=ModelSettings(max_tokens=150),
)

admissions_agent = Agent(
    name="Admissions Counselor",
    instructions="You are a university admissions counselor. Address questions about programs, applications, and document submissions.",
    model=model,
    model_settings=ModelSettings(max_tokens=150),
)

finance_agent = Agent(
    name="Finance Officer",
    instructions="You are a university finance officer. Handle fee disputes, refunds, and billing-related concerns with care.",
    model=model,
    model_settings=ModelSettings(max_tokens=150),
)

main_agent = Agent(
    name="Student Support Assistant",
    instructions="""
    You are the university's first-line support assistant. When a student's concern requires a specialist, 
    pass the case to the correct agent using structured handoffs.

    For academic concerns: issue, subject, deadline impact, previous help attempts.
    For technical problems: error message, device used, browser, urgency, actions tried.
    For admissions: program of interest, documents submitted, application status, questions.
    For fee disputes: component, invoice ID, amount, reason, payment method.
    """,
    model=model,
    model_settings=ModelSettings(max_tokens=150),
    handoffs=[
        handoff(
            agent=academic_agent,
            tool_name_override="academic_concern",
            tool_description_override="Forward academic concern with structured details",
            on_handoff=on_academic_concern_with_data,
            input_type=AcademicConcernData,
        ),
        handoff(
            agent=tech_agent,
            tool_name_override="lms_technical_issue",
            tool_description_override="Transfer LMS or portal issue to tech support",
            on_handoff=on_portal_issue_with_data,
            input_type=PortalIssueData,
        ),
        handoff(
            agent=admissions_agent,
            tool_name_override="admissions_query",
            tool_description_override="Send admissions-related question with details",
            on_handoff=on_admission_query_with_data,
            input_type=AdmissionQueryData,
        ),
        handoff(
            agent=finance_agent,
            tool_name_override="fee_dispute_case",
            tool_description_override="Transfer fee dispute with payment and invoice info",
            on_handoff=on_fee_dispute_with_data,
            input_type=FeeDisputeData,
        ),
    ],
)


# ============================
# Main Demo Function
# ============================
async def main():
    """
    Main demo function for structured handoff scenarios.
    Runs four test cases and prints visually distinct, professional output.
    Returns: None
    """
    print("\n🎓 Student Support Assistant — Structured Handoff Demo\n")
    try:
        # =============================
        # 🧪 Test Case 1: Academic Concern
        # =============================
        print("\n=== Test 1: Academic Concern ===")
        result1 = await Runner.run(
            main_agent,
            input="""I'm really stressed. I'm falling behind in Data Structures because I missed 3 lectures due to illness. \
            There's a project deadline next week, and I already asked in class forum and emailed the TA, but got no reply. \
            I need academic guidance.""",
        )
        print(f"\n🟢 Final Output: {getattr(result1, 'final_output', 'No output')}\n")

        # =============================
        # 🧪 Test Case 2: LMS/Portal Issue
        # =============================
        print("\n=== Test 2: Portal Issue ===")
        result2 = await Runner.run(
            main_agent,
            input="""I can't access the LMS — I keep getting 'Session Timeout' on my mobile phone using Chrome. \
            I’ve already cleared cache, tried another browser, and restarted my phone. This needs fixing ASAP, \
            I have an assignment due tonight!""",
        )
        print(f"\n🟢 Final Output: {getattr(result2, 'final_output', 'No output')}\n")

        # =============================
        # 🧪 Test Case 3: Admissions Query
        # =============================
        print("\n=== Test 3: Admission Query ===")
        result3 = await Runner.run(
            main_agent,
            input="""I'm applying to the BS AI program. I’ve submitted my transcripts and recommendation letter. \
            Can you tell me the status of my application? Also, I want to ask about interview dates and scholarships.""",
        )
        print(f"\n🟢 Final Output: {getattr(result3, 'final_output', 'No output')}\n")

        # =============================
        # 🧪 Test Case 4: Fee Dispute
        # =============================
        print("\n=== Test 4: Fee Dispute ===")
        result4 = await Runner.run(
            main_agent,
            input="""I was charged an extra $150 exam fee this semester. My invoice ID is FEE-2025-00789. \
            I paid using bank transfer. I need this investigated — I never signed up for that exam.""",
        )
        print(f"\n🟢 Final Output: {getattr(result4, 'final_output', 'No output')}\n")

        # =============================
        # ✅ Summary: Benefits of Structured Handoffs
        # =============================
        print("\n=== Benefits of Structured Handoffs ===")
        print(
            """
    ✅ Data Consistency: Prevents missing or misinterpreted context.
    ✅ Validation: Ensures required fields are always provided.
    ✅ Automation: Handoff callbacks can trigger workflows.
    ✅ Contextual Precision: Each agent receives tailored info.
    ✅ Scalable Support: Structure allows for analytics and reporting.
    """
        )
        print("\n🎉 Demo complete!\n")
    except Exception as e:
        print(f"\n❌ An error occurred during the demo: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
