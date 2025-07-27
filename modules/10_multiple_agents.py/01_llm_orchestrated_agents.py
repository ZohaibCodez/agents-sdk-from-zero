"""
modules/10_multiple_agents.py/01.py

Multi-Agent Orchestration Demo for Fitness Planning

This module demonstrates a professional, educational, and robust multi-agent orchestration system for fitness planning using the OpenAI Agents SDK. It features:
- Multiple specialized agents (workout, diet, supplement, orchestrator)
- Tool usage and handoff between agents
- Robust error handling with visually distinct output
- Comprehensive docstrings and inline comments for educational clarity
- Environment variable: GEMINI_API_KEY (required)

Author: Zohaib Khan
References: OpenAI Agents SDK documentation
"""

import asyncio
import os
from pydantic import BaseModel
from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    OpenAIChatCompletionsModel,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    RunResult,
    Runner,
    TResponseInputItem,
    handoff,
    function_tool,
    input_guardrail,
    output_guardrail,
)
from agents.guardrail import GuardrailFunctionOutput
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI

# =========================
# Environment & Model Setup
# =========================
load_dotenv(find_dotenv())

# Configuration constants
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_NAME: str = "gemini-2.0-flash"

# Validate API key
if not GEMINI_API_KEY:
    print("\n❌ [ERROR] GEMINI_API_KEY environment variable is required but not found.\n")
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

print(f"✅ [INFO] Model configured successfully\n")

# =========================
# Tool Definitions
# =========================

@function_tool
def workout_search(goal: str) -> str:
    """Suggest workout plan ideas based on fitness goal.
    Args:
        goal (str): The user's fitness goal (e.g., 'muscle gain', 'fat loss').
    Returns:
        str: A suggested workout plan.
    """
    print(f"[DEBUG] [workout_search] Called with goal: {goal}")
    if "muscle" in goal.lower():
        return (
            """
            Muscle Gain Workout Plan:
            - Day 1: Chest & Triceps (4 sets x 8–12 reps)
            - Day 2: Back & Biceps (4 sets x 8–12 reps)
            - Day 3: Legs (Squats, Deadlifts, Lunges)
            - Day 4: Shoulders & Arms
            Rest Days: 2 per week
            """
        )
    elif "fat" in goal.lower() or "weight loss" in goal.lower():
        return (
            """
            Fat Loss Workout Plan:
            - 4 days/week HIIT + Weight Training
            - Focus: Compound lifts + 20 min cardio after workout
            """
        )
    else:
        return "Please specify muscle gain or fat loss as your fitness goal."

@function_tool
def meal_plan_suggest(calories: int) -> str:
    """Suggest daily meal plans based on calorie needs.
    Args:
        calories (int): Target daily calorie intake.
    Returns:
        str: A sample meal plan.
    """
    print(f"[DEBUG] [meal_plan_suggest] Called with calories: {calories}")
    return f"""
    Sample Meal Plan for {calories} kcal:
    - Breakfast: Oats + Eggs + Fruit
    - Lunch: Chicken Breast + Rice + Vegetables
    - Snacks: Protein Shake + Nuts
    - Dinner: Fish + Sweet Potatoes + Salad
    """

# =========================
# Agent Definitions
# =========================

gym_workout_agent = Agent(
    name="GymWorkoutPlanner",
    instructions="""
    You are a professional workout planner. Your job:
    - Create workout plans based on goals: fat loss, muscle gain, strength.
    - Suggest exercise types, sets, reps.
    - Adjust plans for beginners, intermediate, advanced levels.
    """,
    tools=[workout_search],
    model=model,
)

diet_advisor_agent = Agent(
    name="DietAdvisor",
    instructions="""
    You are a certified diet advisor. Your job:
    - Suggest daily meals for calorie intake.
    - Balance macros: protein, carbs, fats.
    - Adjust diet plans based on workout goals and lifestyle.
    """,
    tools=[meal_plan_suggest],
    model=model,
)

supplement_expert_agent = Agent(
    name="SupplementExpert",
    instructions="""
    You are a supplement consultant. Your job:
    - Recommend safe supplements for fitness goals.
    - Suggest usage guidelines (timing, dosage).
    - Warn about unnecessary or harmful supplements.
    """,
    model=model,
)

fitness_orchestrator = Agent(
    name="FitnessOrchestrator",
    instructions="""
    You are a fitness and health orchestration agent. Your job:
    - Plan complete gym, diet, and supplement strategies for clients.
    - Use tools like workout search and meal plan suggestion.
    - Delegate specialized parts to expert agents:
        - GymWorkoutPlanner
        - DietAdvisor
        - SupplementExpert
    """,
    handoffs=[
        handoff(agent=gym_workout_agent),
        handoff(agent=diet_advisor_agent),
        handoff(agent=supplement_expert_agent),
    ],
    model=model,
)

# =========================
# Demo Functions
# =========================

async def demo_fat_loss_planning():
    """Demo: Fat Loss Fitness Plan Orchestration
    Shows how the orchestrator agent coordinates a fat loss plan.
    """
    print("\n==============================")
    print("🏃‍♂️ [DEMO] Fat Loss Fitness Plan Orchestration")
    print("==============================\n")
    user_request = (
        """
        I want to lose fat. Please help me with:
        1. A structured fat loss workout plan
        2. A calorie-deficit diet plan (around 1800 kcal)
        3. Safe supplement recommendations for fat loss
        """
    )
    try:
        result = await Runner.run(fitness_orchestrator, user_request, max_turns=6)
        print("\n✅ [RESULT] Fat loss plan created successfully!")
        print(f"[OUTPUT]\n{result.final_output}\n")
        return result
    except Exception as e:
        print(f"\n❌ [EXCEPTION] Fat loss planning orchestration failed: {e}\n")
        return None

async def demo_beginner_workout_plan():
    """Demo: Beginner-Friendly Gym Plan
    Shows how the orchestrator agent creates a plan for beginners.
    """
    print("\n==============================")
    print("🧑‍🎓 [DEMO] Beginner-Friendly Gym Plan")
    print("==============================\n")
    user_request = (
        """
        I’m a complete beginner. Please help me with:
        1. A simple workout routine (no complex machines)
        2. A beginner diet plan (2200 kcal)
        3. Supplement guidance for beginners
        """
    )
    try:
        result = await Runner.run(fitness_orchestrator, user_request, max_turns=6)
        print("\n✅ [RESULT] Beginner plan created successfully!")
        print(f"[OUTPUT]\n{result.final_output}\n")
        return result
    except Exception as e:
        print(f"\n❌ [EXCEPTION] Beginner plan orchestration failed: {e}\n")
        return None

async def demo_advanced_strength_program():
    """Demo: Advanced Strength Training Program
    Shows how the orchestrator agent creates a plan for advanced lifters.
    """
    print("\n==============================")
    print("🏋️‍♂️ [DEMO] Advanced Strength Training Program")
    print("==============================\n")
    user_request = (
        """
        I’m an experienced lifter. I want:
        1. A strength-focused gym routine (low rep, high weight)
        2. Diet recommendations for powerlifting (3500 kcal)
        3. Advanced supplement advice (including creatine, etc.)
        """
    )
    try:
        result = await Runner.run(fitness_orchestrator, user_request, max_turns=6)
        print("\n✅ [RESULT] Advanced strength program created successfully!")
        print(f"[OUTPUT]\n{result.final_output}\n")
        return result
    except Exception as e:
        print(f"\n❌ [EXCEPTION] Advanced strength orchestration failed: {e}\n")
        return None

# =========================
# Main Entrypoint
# =========================

async def main():
    """Main entrypoint for running all fitness planning demos."""
    print("\n💪 [MAIN] Gym & Fitness Multi-Agent Orchestration Demo 💪\n")
    # Removed demo_fitness_planning() as it does not exist
    await demo_fat_loss_planning()
    await demo_beginner_workout_plan()
    await demo_advanced_strength_program()
    print("\n✅ [INFO] All demos completed.\n")

if __name__ == "__main__":
    asyncio.run(main())
