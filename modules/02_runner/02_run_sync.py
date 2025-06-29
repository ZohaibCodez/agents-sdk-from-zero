"""
Synchronous Agent Runner Demonstration

This module demonstrates how to use the synchronous runner functionality
to execute agent tasks without async/await patterns. It showcases a
history agent that provides concise answers to user queries.

Author: Zohaib Khan
"""

import os
import asyncio
from typing import Optional
from dotenv import load_dotenv, find_dotenv

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

# Load environment variables and disable tracing for cleaner output
load_dotenv(find_dotenv())
set_tracing_disabled(True)

# Configuration constants
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_NAME: str = "gemini-2.0-flash"

# Validate API key
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required but not found.")

# Initialize external OpenAI client for Gemini
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL,
)

# Configure the chat completions model
model = OpenAIChatCompletionsModel(
    openai_client=external_client, 
    model=GEMINI_MODEL_NAME
)


def create_history_agent() -> Agent:
    """
    Create a specialized agent for providing historical information.
    
    Returns:
        Agent: Configured history agent with concise response instructions.
    """
    return Agent(
        name="History Assistant Agent",
        instructions=(
            "You are a knowledgeable history assistant. Provide concise, "
            "accurate answers about historical events, figures, and periods. "
            "Focus on the most important and relevant information."
        ),
        model=model,
    )


def run_synchronous_agent_demo() -> None:
    """
    Demonstrate synchronous agent execution using Runner.run_sync.
    
    This function creates a history agent and runs a sample query
    to showcase the synchronous execution pattern.
    """
    # Create the history agent
    history_agent = create_history_agent()
    
    # Sample query for demonstration
    sample_query = "Tell me 3 important events of history."
    
    print(f"Query: {sample_query}")
    print("-" * 50)
    
    try:
        # Execute the agent synchronously
        result = Runner.run_sync(
            starting_agent=history_agent,  # Fixed typo: 'startin_agent' -> 'starting_agent'
            input=sample_query,
        )
        
        # Process and display the result
        if result.final_output:
            print(f"[FINAL OUTPUT] - {result.final_output}")
        else:
            print("No output received from the agent.")
            
    except Exception as e:
        print(f"An error occurred during agent execution: {e}")


def main() -> None:
    """
    Main entry point for the synchronous agent demonstration.
    
    Initializes and runs the synchronous agent system.
    """
    print("Starting Synchronous Agent Runner Demo...")
    print("=" * 60)
    
    try:
        run_synchronous_agent_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("\nDemo completed.")


if __name__ == "__main__":
    main()
