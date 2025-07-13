"""
Simple Email Workflow RunHooks - Learning the Basics

This is a simplified version to understand the core RunHooks concepts:
1. Track when agents start and finish
2. Monitor handoffs between agents
3. Log tool usage
4. Collect basic metrics

Focus: Understanding the fundamental lifecycle events
"""

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

# ================================
# 1. Simple Email RunHooks
# ================================


class SimpleEmailHooks(RunHooks):
    """Simple RunHooks to understand the basics."""

    def __init__(self):
        self.events = []  # Store all events
        self.email_count = 0
        self.handoff_count = 0
        self.tool_count = 0

    async def on_agent_start(self, context: Any, agent: Agent) -> None:
        """Called when an agent starts working."""
        self.events.append(f"🟢 {agent.name} started working")
        print(f"🟢 {agent.name} started working on email")

    async def on_agent_end(self, context: Any, agent: Agent, output: Any) -> None:
        """Called when an agent finishes working."""
        self.events.append(f"🔴 {agent.name} finished working")
        self.email_count += 1
        print(f"🔴 {agent.name} finished working on email")

    async def on_handoff(
        self, context: Any, from_agent: Agent, to_agent: Agent
    ) -> None:
        """Called when one agent hands off to another."""
        self.handoff_count += 1
        self.events.append(f"🔄 {from_agent.name} → {to_agent.name}")
        print(f"🔄 Handoff: {from_agent.name} → {to_agent.name}")

    async def on_tool_start(self, context: Any, agent: Agent, tool) -> None:
        """Called when an agent starts using a tool."""
        self.tool_count += 1
        self.events.append(f"🔧 {agent.name} using {tool.name}")
        print(f"🔧 {agent.name} using tool: {tool.name}")

    async def on_tool_end(self, context: Any, agent: Agent, tool, result: str) -> None:
        """Called when an agent finishes using a tool."""
        self.events.append(f"✅ {agent.name} finished {tool.name}")
        print(f"✅ {agent.name} finished using: {tool.name}")

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
    """Check what type of email this is."""
    if "billing" in email_content.lower():
        return "This is a billing question"
    elif "technical" in email_content.lower():
        return "This is a technical issue"
    else:
        return "This is a general inquiry"


@function_tool
def send_reply(message: str) -> str:
    """Send a reply to the customer."""
    return f"Reply sent: {message}"


@function_tool
def look_up_account(customer_email: str) -> str:
    """Look up customer account information."""
    return f"Account found for {customer_email}"


# ================================
# 3. Simple Email Agents
# ================================


def create_email_reader() -> Agent:
    """Agent that reads and categorizes emails."""
    return Agent(
        name="EmailReader",
        instructions="You read emails and figure out what type they are.",
        tools=[check_email_type, look_up_account],
        model=model,
    )


def create_email_responder() -> Agent:
    """Agent that responds to emails."""
    return Agent(
        name="EmailResponder",
        instructions="You write and send replies to customer emails.",
        tools=[send_reply, look_up_account],
        model=model,
    )


def create_specialist() -> Agent:
    """Agent that handles complex issues."""
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
    """Run a simple email workflow to see RunHooks in action."""
    print("=== Simple Email Workflow Demo ===")

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
    result = await Runner.run(
        reader,  # Start with the email reader
        input="Hi, I'm having a technical issue with my billing account",
        hooks=hooks,  # This is where the magic happens!
    )

    print(f"\nFinal Result: {result.final_output}")

    # Show what our hooks captured
    summary = hooks.get_summary()
    print(f"\n📊 What RunHooks Captured:")
    print(f"  Total Events: {summary['total_events']}")
    print(f"  Emails Processed: {summary['emails_processed']}")
    print(f"  Handoffs Made: {summary['handoffs_made']}")
    print(f"  Tools Used: {summary['tools_used']}")

    print(f"\n📋 Step-by-Step Events:")
    for i, event in enumerate(summary["all_events"], 1):
        print(f"  {i}. {event}")


async def main():
    """Run the simple demo."""
    print("📧 Simple Email RunHooks - Learning the Basics")
    print("=" * 50)

    await simple_demo()

    print("\n" + "=" * 50)
    print("✅ Simple demo complete!")
    print("\n🎯 Key Learning Points:")
    print("1. RunHooks 'watch' everything that happens")
    print("2. They capture agent starts, ends, handoffs, and tool usage")
    print("3. You can collect metrics and create reports")
    print("4. This helps you understand how your agents work together")
    print("5. Perfect for debugging and improving your workflows!")


if __name__ == "__main__":
    asyncio.run(main())


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
