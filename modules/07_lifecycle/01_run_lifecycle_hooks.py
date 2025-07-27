"""
modules/07_lifecycle/01_run_lifecycle_hooks.py

Simple Email Workflow RunHooks - Learning the Basics

Description:
    Demonstrates the basics of agent lifecycle hooks (RunHooks) in a simple email workflow using the OpenAI Agents SDK. Tracks agent starts/ends, handoffs, tool usage, and collects basic metrics. Designed for educational clarity and production-readiness.

Features:
    - Agent lifecycle event tracking (start, end, handoff, tool usage)
    - Metrics collection and summary reporting
    - Visually distinct, clearly labeled output with banners and emoji
    - Robust error handling and input validation
    - Educational comments and section headers

Environment Variables:
    - GEMINI_API_KEY: API key for Gemini model (required)

Author:
    Zohaib Khan

References:
    - https://github.com/openai/agents-sdk
    - https://platform.openai.com/docs/agents
    - https://github.com/openai/openai-python
"""

# =========================
# Imports
# =========================
import asyncio
import os
from typing import Any, Dict
from datetime import datetime

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    handoff,
    function_tool,
    set_tracing_disabled,
)
from agents.lifecycle import RunHooks
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI

# =========================
# Environment & Model Setup
# =========================
load_dotenv(find_dotenv())
set_tracing_disabled(True)

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_NAME: str = "gemini-2.0-flash"

if not GEMINI_API_KEY:
    print("\n❌ [ERROR] GEMINI_API_KEY environment variable is required but not found.\n")
    raise ValueError("GEMINI_API_KEY environment variable is required but not found.")

print("\n==============================")
print(f"🚀 Initializing Gemini client with model: {GEMINI_MODEL_NAME}")
print("==============================\n")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL,
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client, model=GEMINI_MODEL_NAME
)

print(f"✅ [INFO] Model configured successfully\n")

# ================================
# 1. Simple Email RunHooks
# ================================

class SimpleEmailHooks(RunHooks):
    """
    Simple RunHooks implementation for educational demonstration.

    Tracks agent lifecycle events, handoffs, and tool usage. Collects metrics and logs events for reporting.
    """
    def __init__(self):
        self.events = []  # Store all events
        self.email_count = 0
        self.handoff_count = 0
        self.tool_count = 0

    async def on_agent_start(self, context: Any, agent: Agent) -> None:
        """Called when an agent starts working."""
        self.events.append(f"🟢 {agent.name} started working")
        print(f"\n🟢 [AGENT START] {agent.name} started working on email\n")

    async def on_agent_end(self, context: Any, agent: Agent, output: Any) -> None:
        """Called when an agent finishes working."""
        self.events.append(f"🔴 {agent.name} finished working")
        self.email_count += 1
        print(f"\n🔴 [AGENT END] {agent.name} finished working on email\n")

    async def on_handoff(
        self, context: Any, from_agent: Agent, to_agent: Agent
    ) -> None:
        """Called when one agent hands off to another."""
        self.handoff_count += 1
        self.events.append(f"🔄 {from_agent.name} → {to_agent.name}")
        print(f"\n🔄 [HANDOFF] {from_agent.name} → {to_agent.name}\n")

    async def on_tool_start(self, context: Any, agent: Agent, tool) -> None:
        """Called when an agent starts using a tool."""
        self.tool_count += 1
        self.events.append(f"🔧 {agent.name} using {tool.name}")
        print(f"\n🔧 [TOOL START] {agent.name} using tool: {tool.name}\n")

    async def on_tool_end(self, context: Any, agent: Agent, tool, result: str) -> None:
        """Called when an agent finishes using a tool."""
        self.events.append(f"✅ {agent.name} finished {tool.name}")
        print(f"\n✅ [TOOL END] {agent.name} finished using: {tool.name}\n")

    def get_summary(self) -> Dict[str, Any]:
        """Get a simple summary of what happened."""
        return {
            "total_events": len(self.events),
            "emails_processed": self.email_count,
            "handoffs_made": self.handoff_count,
            "tools_used": self.tool_count,
            "all_events": self.events,
        }

# ================================
# 2. Simple Email Tools
# ================================

@function_tool
def check_email_type(email_content: str) -> str:
    """
    Check what type of email this is.

    Args:
        email_content (str): The content of the email.
    Returns:
        str: The type of email (billing, technical, or general inquiry).
    """
    if "billing" in email_content.lower():
        return "This is a billing question"
    elif "technical" in email_content.lower():
        return "This is a technical issue"
    else:
        return "This is a general inquiry"

@function_tool
def send_reply(message: str) -> str:
    """
    Send a reply to the customer.

    Args:
        message (str): The reply message to send.
    Returns:
        str: Confirmation that the reply was sent.
    """
    return f"Reply sent: {message}"

@function_tool
def look_up_account(customer_email: str) -> str:
    """
    Look up customer account information.

    Args:
        customer_email (str): The customer's email address.
    Returns:
        str: Account lookup result.
    """
    return f"Account found for {customer_email}"

# ================================
# 3. Simple Email Agents
# ================================

def create_email_reader() -> Agent:
    """
    Create the EmailReader agent.

    Returns:
        Agent: Configured EmailReader agent.
    """
    return Agent(
        name="EmailReader",
        instructions="You read emails and figure out what type they are.",
        tools=[check_email_type, look_up_account],
        model=model,
    )

def create_email_responder() -> Agent:
    """
    Create the EmailResponder agent.

    Returns:
        Agent: Configured EmailResponder agent.
    """
    return Agent(
        name="EmailResponder",
        instructions="You write and send replies to customer emails.",
        tools=[send_reply, look_up_account],
        model=model,
    )

def create_specialist() -> Agent:
    """
    Create the Specialist agent for complex issues.

    Returns:
        Agent: Configured Specialist agent.
    """
    return Agent(
        name="Specialist",
        instructions="You handle complex technical or billing issues.",
        tools=[send_reply, look_up_account],
        model=model,
    )

# ================================
# 4. Simple Demo
# ================================

async def simple_demo():
    """
    Run a simple email workflow to see RunHooks in action.
    """
    print("\n==============================")
    print("📧 [DEMO] Simple Email Workflow Demo")
    print("==============================\n")

    try:
        # Create agents
        reader = create_email_reader()
        responder = create_email_responder()
        specialist = create_specialist()

        # Set up handoffs (who can pass work to whom)
        reader.handoffs = [
            handoff(responder, tool_name_override="send_to_responder"),
            handoff(specialist, tool_name_override="send_to_specialist"),
        ]

        responder.handoffs = [
            handoff(specialist, tool_name_override="escalate_to_specialist")
        ]

        # Create our simple hooks
        hooks = SimpleEmailHooks()

        # Process an email
        print("\n📝 [INPUT] Email: 'Hi, I'm having a technical issue with my billing account'\n")
        result = await Runner.run(
            reader,  # Start with the email reader
            input="Hi, I'm having a technical issue with my billing account",
            hooks=hooks,  # This is where the magic happens!
        )

        print(f"\n🏁 [FINAL OUTPUT] {result.final_output}\n")

        # Show what our hooks captured
        summary = hooks.get_summary()
        print(f"\n📊 [SUMMARY] What RunHooks Captured:")
        print(f"  Total Events: {summary['total_events']}")
        print(f"  Emails Processed: {summary['emails_processed']}")
        print(f"  Handoffs Made: {summary['handoffs_made']}")
        print(f"  Tools Used: {summary['tools_used']}")

        print(f"\n📋 [EVENT LOG] Step-by-Step Events:")
        for i, event in enumerate(summary["all_events"], 1):
            print(f"  {i}. {event}")

    except Exception as e:
        print(f"\n❌ [ERROR] Exception during demo: {e}\n")

async def main():
    """
    Run the simple demo and print key learning points.
    """
    print("\n==================================================")
    print("📧 Simple Email RunHooks - Learning the Basics")
    print("==================================================\n")

    await simple_demo()

    print("\n==================================================")
    print("✅ [COMPLETE] Simple demo complete!")
    print("==================================================\n")
    print("🎯 [KEY LEARNING POINTS]")
    print("1. RunHooks 'watch' everything that happens")
    print("2. They capture agent starts, ends, handoffs, and tool usage")
    print("3. You can collect metrics and create reports")
    print("4. This helps you understand how your agents work together")
    print("5. Perfect for debugging and improving your workflows!\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ [ERROR] Unhandled exception in main: {e}\n")

# ================================
# 5. What You Just Learned
# ================================

"""
CORE CONCEPTS MASTERED:

1. **RunHooks Class**: 
   - Inherits from RunHooks
   - Implements the 5 key methods (on_agent_start, on_agent_end, etc.)

2. **Event Tracking**:
   - Every action gets logged
   - You can see the complete workflow

3. **Metrics Collection**:
   - Count emails, handoffs, tools used
   - Build reports from the data

4. **Real-time Monitoring**:
   - See what's happening as it happens
   - Perfect for debugging workflows

5. **Simple Implementation**:
   - Just store events in a list
   - Print to console for immediate feedback
   - Get summary at the end

Next Steps:
- Try adding more metrics (timing, success rates)
- Create different types of hooks for different needs
- Experiment with filtering events
- Build dashboards from the collected data
"""
