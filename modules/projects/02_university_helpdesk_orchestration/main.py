"""
University Helpdesk Orchestration
--------------------------------
A multi-agent orchestration system simulating a university helpdesk. This project demonstrates advanced agent handoffs, context-aware routing, and tool usage for handling student queries across multiple departments (helpdesk, finance, IT, academic advising, admissions, student affairs).

Features:
- Modular agents for each department, each with specialized instructions and tools
- Orchestrator agent that intelligently routes queries based on student type, urgency, and context
- Realistic handoff and escalation logic between agents
- Streaming output of agent/tool actions and handoff events
- Example test cases simulating various student scenarios

Environment:
- Requires MISTRAL_API_KEY for LLM access
- Designed for educational and demonstration purposes
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
from agents.extensions.handoff_prompt import (
    RECOMMENDED_PROMPT_PREFIX,
    prompt_with_handoff_instructions,
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

MISTRAL_API_KEY: str | None = os.getenv("MISTRAL_API_KEY")
MODEL_NAME: str = "mistral/mistral-small-2506"

if not MISTRAL_API_KEY:
    print("❌ MISTRAL_API_KEY environment variable is required but not found.")
    raise ValueError("MISTRAL_API_KEY environment variable is required but not found.")

print(f"🚀 Initializing OpenRouter client with model: {MODEL_NAME}")
model = LitellmModel(model=MODEL_NAME, api_key=MISTRAL_API_KEY)
print(f"✅ Model configured successfully\n")

# =============================================================================
# CONTEXT MODELS
# =============================================================================


class StudentType(str, Enum):
    """Enumeration of possible student types for routing and context-aware logic."""

    FRESHMAN = "freshman"
    SENIOR = "senior"
    GRADUATE = "graduate"
    INTERNATIONAL = "international"


class QueryUrgency(str, Enum):
    """Enumeration of urgency levels for student queries."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HandoffContext(BaseModel):
    """
    Context object passed between agents during handoff.

    Attributes:
        student_type (StudentType): The type of student (freshman, senior, etc.)
        query_urgency (QueryUrgency): The urgency of the query
        escalation_count (int): Number of times the case has been escalated
        previous_agents (list[str]): List of agent names that have handled the case
        resolution_attempts (list[str]): List of attempted resolutions
        student_feedback (Optional[str]): Feedback from the student, if any
    """

    student_type: StudentType
    query_urgency: QueryUrgency
    escalation_count: int = 0
    previous_agents: list[str] = Field(default_factory=list)
    resolution_attempts: list[str] = Field(default_factory=list)
    student_feedback: Optional[str] = None


# =============================================================================
# TOOLS
# =============================================================================


@function_tool
def lookup_student_type(student_id: str) -> str:
    """
    Look up the student type based on their ID.

    Args:
        student_id (str): The unique student identifier.
    Returns:
        str: The student type (e.g., 'freshman', 'senior', etc.)
    """
    return {
        "stu_001": "international",
        "stu_002": "graduate",
        "stu_003": "freshman",
        "stu_004": "senior",
    }.get(student_id, "freshman")


@function_tool
def analyze_query_urgency(query: str) -> str:
    """
    Analyze the urgency of a student's query based on keywords.

    Args:
        query (str): The student's query text.
    Returns:
        str: The urgency level ('critical', 'high', 'medium', 'low').
    """
    if "urgent" in query or "crash" in query or "blocked" in query:
        return "critical"
    elif "payment" in query or "fee" in query:
        return "high"
    elif "course" in query or "enroll" in query:
        return "medium"
    else:
        return "low"


@function_tool
def log_helpdesk_interaction(agent_name: str, action: str, student_id: str) -> str:
    """
    Log an interaction performed by an agent for a student.

    Args:
        agent_name (str): The name of the agent.
        action (str): The action taken.
        student_id (str): The student's ID.
    Returns:
        str: Log message.
    """
    return f"LOGGED: {agent_name} did '{action}' for student {student_id}"


@function_tool
def escalate_to_department_head(details: str, student_type: str) -> str:
    """
    Escalate an issue to the department head.

    Args:
        details (str): Details of the issue.
        student_type (str): The type of student involved.
    Returns:
        str: Escalation message.
    """
    return f"Escalated to Dept Head: {details} (Student Type: {student_type})"


@function_tool
def create_support_ticket(issue: str, priority: str) -> str:
    """
    Create a support ticket for an issue.

    Args:
        issue (str): Description of the issue.
        priority (str): Priority level.
    Returns:
        str: Ticket creation message.
    """
    return f"Support Ticket Created: [{priority}] - {issue}"


# =============================================================================
# AGENT DEFINITIONS WITH IMPROVED PROMPTS
# =============================================================================


def create_helpdesk_agent():
    """
    Create the Helpdesk Support agent with specialized instructions and tools.
    Returns:
        Agent: Configured Helpdesk agent.
    """
    return Agent(
        name="Helpdesk Support",
        instructions=prompt_with_handoff_instructions(
            """
You are the First-Line Helpdesk Support Agent for the University Student Support System.

ROLE & RESPONSIBILITIES:
- Serve as the primary point of contact for all student inquiries
- Provide immediate assistance and emotional support to students
- Accurately triage and route complex issues to specialized departments
- Maintain detailed records of all student interactions

INTERACTION GUIDELINES:
1. Always greet students warmly and professionally
2. Listen actively and acknowledge their concerns with empathy
3. Ask clarifying questions to fully understand the issue
4. Provide clear, step-by-step guidance when possible
5. If unable to resolve immediately, explain next steps clearly

DECISION MAKING PROCESS:
- For simple queries: Provide direct assistance and resolution
- For complex issues: Gather all necessary information before handoff
- For urgent matters: Prioritize immediate response and escalation
- Always use lookup_student_type and analyze_query_urgency tools first

HANDOFF CRITERIA:
- Finance issues → Finance Office
- Academic problems → Academic Advisor  
- Technical issues → IT Helpdesk
- Admissions queries → Admissions Counselor
- Unresolved/complex cases → Student Affairs

Remember: Your goal is to make every student feel heard, supported, and confident that their issue will be resolved efficiently.
        """
        ),
        tools=[lookup_student_type, analyze_query_urgency, log_helpdesk_interaction],
        model=model,
    )


def create_finance_agent():
    """
    Create the Finance Office agent with specialized instructions and tools.
    Returns:
        Agent: Configured Finance Office agent.
    """
    return Agent(
        name="Finance Office",
        instructions=prompt_with_handoff_instructions(
            """
You are the Finance Office Specialist for the University Student Support System.

EXPERTISE AREAS:
- Tuition and fee management
- Payment processing and troubleshooting
- Financial aid and scholarship administration
- Billing discrepancies and refunds
- Payment plan setup and modifications

APPROACH:
1. Immediately acknowledge any financial stress the student may be experiencing
2. Verify student identity and account details carefully
3. Explain financial processes in simple, understandable terms
4. Provide multiple payment options when available
5. Document all financial interactions for audit purposes

SPECIFIC PROTOCOLS:
- Payment failures: Investigate transaction details, verify bank involvement
- Fee disputes: Review billing history, check for system errors
- Late payments: Assess penalty waivers, discuss payment plan options
- Financial aid: Verify eligibility, explain disbursement timelines

ESCALATION RULES:
- Complex financial aid cases → Student Affairs
- System-wide billing issues → University IT Lead
- Policy exceptions requiring approval → Department Head

Always maintain confidentiality and handle financial information with extreme care.
        """
        ),
        tools=[log_helpdesk_interaction, create_support_ticket],
        model=model,
    )


def create_advisor_agent():
    """
    Create the Academic Advisor agent with specialized instructions and tools.
    Returns:
        Agent: Configured Academic Advisor agent.
    """
    return Agent(
        name="Academic Advisor",
        instructions=prompt_with_handoff_instructions(
            """
You are the Academic Advisor for the University Student Support System.

CORE RESPONSIBILITIES:
- Academic planning and course selection guidance
- Enrollment and registration support
- Degree requirement verification
- Academic policy interpretation
- Student success planning and intervention

ADVISORY APPROACH:
1. Understand the student's academic goals and career aspirations
2. Review their academic history and current standing
3. Provide personalized recommendations based on their situation
4. Explain academic policies clearly and their implications
5. Empower students to make informed decisions about their education

COMMON SCENARIOS:
- Course conflicts: Analyze schedule options, suggest alternatives
- Prerequisite issues: Verify requirements, explore waiver possibilities
- Registration problems: Identify system or policy barriers
- Academic planning: Create semester-by-semester roadmaps
- Policy questions: Interpret regulations in student-friendly language

COLLABORATION POINTS:
- Technical registration issues → IT Helpdesk
- Financial barriers to enrollment → Finance Office
- Complex academic appeals → Student Affairs

Focus on being a mentor who guides students toward academic success while ensuring they understand their options and responsibilities.
        """
        ),
        tools=[log_helpdesk_interaction, create_support_ticket],
        model=model,
    )


def create_it_helpdesk_agent():
    """
    Create the IT Helpdesk agent with specialized instructions and tools.
    Returns:
        Agent: Configured IT Helpdesk agent.
    """
    return Agent(
        name="IT Helpdesk",
        instructions=prompt_with_handoff_instructions(
            """
You are the IT Helpdesk Specialist for the University Student Support System.

TECHNICAL EXPERTISE:
- Learning Management System (LMS) support
- Student portal and account access
- Email and communication platform setup
- Software licensing and installation
- Network connectivity troubleshooting

TROUBLESHOOTING METHODOLOGY:
1. Gather specific error messages and system information
2. Reproduce the issue when possible
3. Apply standard troubleshooting steps systematically
4. Document solutions for future reference
5. Verify resolution with the student before closing

PRIORITY HANDLING:
- CRITICAL: Exam access issues, system crashes during deadlines
- HIGH: Login failures, email access problems
- MEDIUM: Software installation, minor functionality issues
- LOW: General questions, enhancement requests

ESCALATION TRIGGERS:
- System-wide outages → University IT Lead
- Security-related incidents → University IT Lead
- Infrastructure issues → University IT Lead
- Policy violations → Student Affairs

Always explain technical solutions in non-technical terms and provide step-by-step instructions the student can follow.
        """
        ),
        tools=[
            log_helpdesk_interaction,
            escalate_to_department_head,
            create_support_ticket,
        ],
        model=model,
    )


def create_it_lead_agent():
    """
    Create the University IT Lead agent with specialized instructions and tools.
    Returns:
        Agent: Configured University IT Lead agent.
    """
    return Agent(
        name="University IT Lead",
        instructions=prompt_with_handoff_instructions(
            """
You are the University IT Lead responsible for high-level technical decision-making and system-wide issues.

LEADERSHIP RESPONSIBILITIES:
- System architecture and infrastructure oversight
- Security incident response and coordination
- Cross-departmental technical integration
- Strategic technology planning and implementation
- Vendor management and technical partnerships

ESCALATION HANDLING:
1. Assess the scope and impact of technical issues
2. Coordinate with multiple departments when necessary
3. Make decisions about system maintenance and upgrades
4. Communicate technical issues to non-technical stakeholders
5. Ensure compliance with university IT policies

DECISION AUTHORITY:
- System-wide maintenance windows
- Security protocol implementations
- Emergency technical responses
- Resource allocation for critical issues
- Policy exceptions for technical requirements

COMMUNICATION STYLE:
- Provide clear timelines and expectations
- Translate technical complexity into business impact
- Maintain transparency about system limitations
- Coordinate with administration on major decisions

You have the authority to make executive technical decisions and coordinate resources across the university.
        """
        ),
        tools=[
            log_helpdesk_interaction,
            escalate_to_department_head,
            create_support_ticket,
        ],
        model=model,
    )


def create_student_affairs_agent():
    """
    Create the Student Affairs agent with specialized instructions and tools.
    Returns:
        Agent: Configured Student Affairs agent.
    """
    return Agent(
        name="Student Affairs",
        instructions=prompt_with_handoff_instructions(
            """
You are the Student Affairs Specialist, handling the most complex and sensitive student cases.

SPECIALIZED ROLE:
- Final escalation point for unresolved issues
- Student advocacy and crisis intervention
- Policy interpretation and exception requests
- Interdepartmental coordination for complex cases
- Student complaint resolution and mediation

CASE MANAGEMENT APPROACH:
1. Thoroughly review the complete case history
2. Identify all stakeholders and their concerns
3. Develop comprehensive resolution strategies
4. Coordinate with multiple departments when needed
5. Ensure follow-up and student satisfaction

AUTHORITY LEVEL:
- Policy exception approvals
- Interdepartmental coordination
- Executive decision-making for student welfare
- Crisis intervention and emergency response
- Final appeal resolution

SENSITIVE SITUATIONS:
- Academic integrity violations
- Personal emergencies affecting academics
- Discrimination or harassment reports
- Financial hardship cases
- Complex multi-departmental issues

COMMUNICATION PRINCIPLES:
- Maintain strict confidentiality
- Show genuine concern for student welfare
- Provide clear explanations of decisions and processes
- Ensure students understand their rights and options
- Document all interactions thoroughly

You serve as the student's strongest advocate within the university system while ensuring institutional policies are followed appropriately.
        """
        ),
        tools=[log_helpdesk_interaction, create_support_ticket],
        model=model,
    )


def create_admissions_agent():
    """
    Create the Admissions Counselor agent with specialized instructions and tools.
    Returns:
        Agent: Configured Admissions Counselor agent.
    """
    return Agent(
        name="Admissions Counselor",
        instructions=prompt_with_handoff_instructions(
            """
You are the Admissions Counselor specializing in student enrollment and program transitions.

ADMISSIONS EXPERTISE:
- New student application guidance
- Program change and transfer processes
- Credit transfer evaluation and approval
- Admission requirement interpretation
- Enrollment deadline and process management

COUNSELING APPROACH:
1. Understand the student's academic and career goals
2. Assess their current academic standing and background
3. Explain admission requirements and processes clearly
4. Guide them through application and documentation steps
5. Provide realistic timelines and expectations

PROCESS MANAGEMENT:
- Application review and status updates
- Document verification and requirements
- Prerequisite assessment and planning
- Transfer credit evaluation coordination
- Enrollment confirmation and next steps

SPECIALIZED SCENARIOS:
- Program upgrades (Bachelor's to Master's, etc.)
- Internal transfers between departments
- International student admission requirements
- Non-traditional student pathways
- Academic fresh start programs

COLLABORATION REQUIREMENTS:
- Academic eligibility → Academic Advisor
- Financial aid implications → Finance Office
- Technical application issues → IT Helpdesk
- Complex policy cases → Student Affairs

Focus on making the admissions process as smooth and transparent as possible while ensuring all requirements are met.
        """
        ),
        tools=[log_helpdesk_interaction],
        model=model,
    )


# =============================================================================
# ESCALATION TRACKING CALLBACKS
# =============================================================================


async def on_escalation_tracking(
    ctx: RunContextWrapper[None], handoff_data: HandoffContext
):
    """
    Callback for tracking escalations between agents.
    Increments escalation count and prints escalation path.

    Args:
        ctx (RunContextWrapper[None]): The run context.
        handoff_data (HandoffContext): The handoff context object.
    """
    handoff_data.escalation_count += 1
    print(f"🔁 Escalation #{handoff_data.escalation_count}")
    print(f"   → {' → '.join(handoff_data.previous_agents)}")


async def on_student_routing(
    ctx: RunContextWrapper[None], handoff_data: HandoffContext
):
    """
    Callback for smart routing events.
    Prints student type, urgency, escalation count, and previous agents.

    Args:
        ctx (RunContextWrapper[None]): The run context.
        handoff_data (HandoffContext): The handoff context object.
    """
    print(f"📌 SMART ROUTING:")
    print(f"   Student Type: {handoff_data.student_type}")
    print(f"   Query Urgency: {handoff_data.query_urgency}")
    print(f"   Escalation Count: {handoff_data.escalation_count}")
    print(f"   Previous Agents: {' → '.join(handoff_data.previous_agents)}")


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================


async def main():
    """
    Main orchestration function for the University Helpdesk demo.
    Sets up all agents, orchestrator, and runs demo/test cases.
    """
    # Agents
    helpdesk = create_helpdesk_agent()
    finance = create_finance_agent()
    advisor = create_advisor_agent()
    it_helpdesk = create_it_helpdesk_agent()
    it_lead = create_it_lead_agent()
    affairs = create_student_affairs_agent()
    admissions = create_admissions_agent()

    # Orchestrator with improved prompt
    orchestrator = Agent(
        name="University Helpdesk Orchestrator",
        instructions=(
            """
You are the University Helpdesk Orchestrator, responsible for intelligently routing student queries to the most appropriate department.

CORE MISSION:
Ensure every student receives prompt, accurate, and empathetic support by connecting them with the right specialist immediately.

ROUTING INTELLIGENCE:
1. Analyze student type (freshman, senior, graduate, international) for context-aware routing
2. Assess query urgency (low, medium, high, critical) to prioritize appropriately
3. Consider previous interaction history to avoid circular routing
4. Balance workload across departments while maintaining service quality

ROUTING DECISION MATRIX:
- FIRST CONTACT: Always route to Helpdesk Support for triage
- FINANCIAL ISSUES: Finance Office (payments, fees, scholarships)
- ACADEMIC MATTERS: Academic Advisor (courses, enrollment, planning)
- TECHNICAL PROBLEMS: IT Helpdesk (LMS, login, software)
- ADMISSIONS QUERIES: Admissions Counselor (applications, transfers)
- ESCALATIONS: IT Lead (system issues) or Student Affairs (complex cases)

SPECIAL CONSIDERATIONS:
- International students may need additional cultural sensitivity
- Graduate students often have complex interdepartmental issues
- Freshman require more guidance and explanation
- Senior students need efficient, direct resolution

ESCALATION TRIGGERS:
- Multiple failed resolution attempts
- Cross-departmental coordination required
- Policy exceptions needed
- Student dissatisfaction expressed
- Critical urgency with time constraints

Always prioritize student satisfaction and efficient resolution over departmental boundaries.
        """
        ),
        tools=[lookup_student_type, analyze_query_urgency, log_helpdesk_interaction],
        model=model,
        handoffs=[
            handoff(
                agent=helpdesk,
                tool_name_override="route_to_helpdesk",
                tool_description_override="Route to first-line support for initial triage and basic assistance",
                on_handoff=on_student_routing,
                input_type=HandoffContext,
            ),
            handoff(
                agent=finance,
                tool_name_override="route_to_finance",
                tool_description_override="Route to Finance Office for payment, billing, and financial aid issues",
                on_handoff=on_student_routing,
                input_type=HandoffContext,
            ),
            handoff(
                agent=advisor,
                tool_name_override="route_to_advisor",
                tool_description_override="Route to Academic Advisor for course, enrollment, and degree planning issues",
                on_handoff=on_student_routing,
                input_type=HandoffContext,
            ),
            handoff(
                agent=it_helpdesk,
                tool_name_override="route_to_it",
                tool_description_override="Route to IT Helpdesk for technical issues and system access problems",
                on_handoff=on_student_routing,
                input_type=HandoffContext,
            ),
            handoff(
                agent=admissions,
                tool_name_override="route_to_admissions",
                tool_description_override="Route to Admissions for application, transfer, and program change queries",
                on_handoff=on_student_routing,
                input_type=HandoffContext,
            ),
            # Escalations
            handoff(
                agent=it_lead,
                tool_name_override="escalate_to_it_lead",
                tool_description_override="Escalate to IT Lead for system-wide or complex technical issues",
                on_handoff=on_escalation_tracking,
                input_type=HandoffContext,
            ),
            handoff(
                agent=affairs,
                tool_name_override="escalate_to_student_affairs",
                tool_description_override="Escalate to Student Affairs for complex, sensitive, or unresolved cases",
                on_handoff=on_escalation_tracking,
                input_type=HandoffContext,
            ),
        ],
    )

    # Simulations
    print("\n=== University Helpdesk Orchestration ===\n")

    # --- DEMO/TEST CASES ---
    inputs = [
        (
            "International student urgent LMS issue",
            """
        Student ID: stu_001
        Query: I urgently need to access the LMS. It's blocked and my final exam is today.
        """,
        ),
        (
            "Graduate student payment failure",
            """
        Student ID: stu_002
        Query: My tuition fee payment failed but the amount was deducted from my bank.
        """,
        ),
        (
            "Freshman course selection help",
            """
        Student ID: stu_003
        Query: I want to change my elective but the portal doesn't allow it.
        """,
        ),
        (
            "Senior with technical + finance issue",
            """
        Student ID: stu_004
        Query: I can't access my result portal, and my fees show incorrect dues.
        """,
        ),
        (
            "New student asking for admission info",
            """
        Student ID: stu_005
        Query: I'm interested in switching from BSCS to BSE. What's the process?
        """,
        ),
    ]

    for title, input_text in inputs:
        print(f"--- {title} ---")
        result = Runner.run_streamed(orchestrator, input=input_text)
        current_agent: str = "Unknown"

        async for event in result.stream_events():
            if event.type == "raw_response_event":
                continue
            elif event.type == "agent_updated_stream_event":
                current_agent = event.new_agent.name
                print(f"🔄 Agent switched to: {current_agent}")
            elif event.type == "run_item_stream_event":
                item = event.item
                if item.type == "handoff_call_item":
                    print(f"🤝 Handing off ... ")
                elif item.type == "handoff_output_item":
                    print(
                        f"📤 Switching from {item.source_agent.name} to {item.target_agent.name}"
                    )
                elif item.type == "tool_call_item":
                    # Fix for linter error - use getattr to safely access name attribute
                    tool_name = getattr(item.raw_item, "name", "Unknown Tool")
                    print(f"🔧 Tool called: {tool_name}")
                elif item.type == "tool_call_output_item":
                    print(f"📊 Tool Output: {item.output}")
                elif item.type == "message_output_item":
                    message_text = ItemHelpers.text_message_output(item)
                    print(f"💬 {current_agent} says: {message_text}")
        print(f"Final Output: {result.final_output}\n")


if __name__ == "__main__":
    asyncio.run(main())
