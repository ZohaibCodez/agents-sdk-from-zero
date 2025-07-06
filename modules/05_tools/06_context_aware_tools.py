"""
06_context_aware_tools.py

Context-Aware Tools Example
https://openai.github.io/openai-agents-python/tools/

This module demonstrates how to create context-aware function tools that can access
user session information, usage statistics, and provide personalized responses.

Key Features:
- User session context with user_id, project_name, and preferred_language
- Tools that access context information for personalization
- Usage statistics tracking
- Multilingual support with user preferences
- Chain multiple tools together for complex workflows

Environment variables:
    - GEMINI_API_KEY: Required for Gemini model access.

Author: Zohaib Khan
"""

import asyncio
import os
from typing import Any
from dotenv import load_dotenv, find_dotenv

from openai import AsyncOpenAI
from agents.handoffs import Handoff, handoff
from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    RunResultStreaming,
    set_tracing_disabled,
    function_tool,
    RunContextWrapper,
    ItemHelpers,
)
from dataclasses import dataclass

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


@dataclass
class UserSessionContext:
    """User session context for personalizing tool responses."""
    user_id: str
    project_name: str
    preferred_language: str


@function_tool
def summarize_note(ctx: RunContextWrapper[UserSessionContext], note: str) -> str:
    """
    Summarize a note with personalized context.
    
    Args:
        ctx: Run context containing user session information
        note: The note to summarize
        
    Returns:
        A personalized summary of the note
    """
    try:
        user: str = ctx.context.user_id if ctx.context else "unknown"
        lang: str = ctx.context.preferred_language if ctx.context else "unknown"
        project: str = ctx.context.project_name if ctx.context else "unknown"

        # Create a simple summary (first sentence or first 40 chars)
        summary: str = note.split(".")[0] + "." if "." in note else note[:40] + "..."
        
        return f"[Summary for {user} working on '{project}' in {lang}]\nSummary: {summary}"
    except Exception as e:
        return f"❌ Error summarizing note: {str(e)}"


@function_tool
def track_usage_stats(ctx: RunContextWrapper[UserSessionContext]) -> str:
    """
    Track and report token usage statistics for the current session.
    
    Args:
        ctx: Run context containing user session information and usage stats
        
    Returns:
        A formatted report of token usage
    """
    try:
        usage = ctx.usage

        # Safely extract usage statistics
        prompt: str | int = getattr(usage, 'input_tokens', '?') if usage else "?"
        completion: str | int = getattr(usage, 'output_tokens', '?') if usage else "?"
        total: str | int = getattr(usage, 'total_tokens', '?') if usage else "?"

        user: str = ctx.context.user_id if ctx.context else "unknown"
        project: str = ctx.context.project_name if ctx.context else "unknown"

        return (
            f"📊 Token usage report for {user} in project '{project}':\n"
            f"🧠 Input Tokens: {prompt}\n"
            f"🗣️ Output Tokens: {completion}\n"
            f"🔢 Total Tokens: {total}"
        )
    except Exception as e:
        return f"❌ Error tracking usage stats: {str(e)}"


@function_tool
def get_project_info(ctx: RunContextWrapper[UserSessionContext]) -> str:
    """
    Get current project and user session information.
    
    Args:
        ctx: Run context containing user session information
        
    Returns:
        A formatted display of current project information
    """
    try:
        if not ctx.context:
            return "❌ No user session context available"

        user: str = ctx.context.user_id if ctx.context else "unknown"
        lang: str = ctx.context.preferred_language if ctx.context else "unknown"
        project: str = ctx.context.project_name if ctx.context else "unknown"

        return (
            f"👤 User: {user}\n"
            f"📁 Project: {project}\n"
            f"🌐 Preferred Language: {lang}\n"
            f"You're currently working on '{project}' and your preferred language is {lang}."
        )
    except Exception as e:
        return f"❌ Error getting project info: {str(e)}"


@function_tool
def translate_note(ctx: RunContextWrapper[UserSessionContext], note: str) -> str:
    """
    Translate a note to the user's preferred language.
    
    Args:
        ctx: Run context containing user session information
        note: The note to translate
        
    Returns:
        A translated version of the note (simulated)
    """
    try:
        user: str = ctx.context.user_id if ctx.context else "unknown"
        lang: str = ctx.context.preferred_language if ctx.context else "English"

        # Simulate translation (reverse the text for demo purposes)
        fake_translation = f"[{lang} version of the note]: {note[::-1]}"

        return (
            f"🌍 Translation for user '{user}' (Language: {lang}):\n{fake_translation}"
        )
    except Exception as e:
        return f"❌ Error translating note: {str(e)}"


# =========================
# Agent Definition
# =========================

smart_agent: Agent[UserSessionContext] = Agent[UserSessionContext](
    name="Smart Data Agent",
    instructions="""
    You are SmartDataAgent — a personalized assistant that can:
    1. Summarize notes with user context
    2. Report token usage statistics
    3. Provide project session information
    4. Translate notes into the user's preferred language

    Always personalize your answers using the user's context.
    When asked to perform multiple tasks, use the appropriate tools in sequence.
    """,
    tools=[summarize_note, track_usage_stats, get_project_info, translate_note],
    model=model,
)


# =========================
# Main Execution
# =========================

async def main():
    """
    Demonstrate context-aware tools with personalized user sessions.
    
    Tests various scenarios including:
    - Note summarization with user context
    - Usage statistics tracking
    - Project information retrieval
    - Note translation to preferred language
    - Chaining multiple tools together
    """
    try:
        # ✅ 1. Create user session context
        context = UserSessionContext(
            user_id="zohaib123", 
            project_name="AI Research Logs", 
            preferred_language="Urdu"
        )

        print("\n" + "="*60)
        print("🧠 CONTEXT-AWARE TOOLS DEMO")
        print("="*60)
        print(f"👤 User: {context.user_id}")
        print(f"📁 Project: {context.project_name}")
        print(f"🌐 Language: {context.preferred_language}")
        print("="*60)

        # Test 1: Summarize Note
        print("\n" + "🔍 TEST 1: Summarize Note\n" + "="*40)
        input_text = "Summarize this note: Today I studied context-aware tools and they were very helpful in building session-aware agents."
        print(f"📥 INPUT: {input_text}")
        print("-" * 60)
        
        result1 = await Runner.run(
            smart_agent,
            input=input_text,
            context=context,
        )
        print(f"📤 OUTPUT: {result1.final_output}")
        print("✅ Test 1 completed successfully!")

        # Test 2: Track Token Usage
        print("\n" + "📊 TEST 2: Track Token Usage\n" + "="*35)
        input_text = "How many tokens have we used so far?"
        print(f"📥 INPUT: {input_text}")
        print("-" * 60)
        
        result2 = await Runner.run(
            smart_agent, 
            input=input_text, 
            context=context
        )
        print(f"📤 OUTPUT: {result2.final_output}")
        print("✅ Test 2 completed successfully!")

        # Test 3: Get Project Info
        print("\n" + "📁 TEST 3: Get Project Info\n" + "="*35)
        input_text = "Can you show me my current project information?"
        print(f"📥 INPUT: {input_text}")
        print("-" * 60)
        
        result3 = await Runner.run(
            smart_agent,
            input=input_text,
            context=context,
        )
        print(f"📤 OUTPUT: {result3.final_output}")
        print("✅ Test 3 completed successfully!")

        # Test 4: Translate Note
        print("\n" + "🌐 TEST 4: Translate Note\n" + "="*35)
        input_text = "Translate this note: Multilingual support is important for global tools."
        print(f"📥 INPUT: {input_text}")
        print("-" * 60)
        
        result4 = await Runner.run(
            smart_agent,
            input=input_text,
            context=context,
        )
        print(f"📤 OUTPUT: {result4.final_output}")
        print("✅ Test 4 completed successfully!")

        # Test 5: Chain Tools
        print("\n" + "🔀 TEST 5: Chain Tools (Summarize + Translate)\n" + "="*25)
        input_text = "Summarize this: I implemented context-aware agents today. Then translate it to my preferred language."
        print(f"📥 INPUT: {input_text}")
        print("-" * 60)
        
        result5 = await Runner.run(
            smart_agent,
            input=input_text,
            context=context,
        )
        print(f"📤 OUTPUT: {result5.final_output}")
        print("✅ Test 5 completed successfully!")

        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📋 Summary:")
        print("   ✅ Note summarization with user context")
        print("   ✅ Token usage tracking")
        print("   ✅ Project information retrieval")
        print("   ✅ Note translation to preferred language")
        print("   ✅ Multi-tool chaining")
        print("="*60)

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
