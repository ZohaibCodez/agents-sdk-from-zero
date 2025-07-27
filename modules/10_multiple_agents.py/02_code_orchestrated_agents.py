"""
modules/10_multiple_agents.py/02_code_orchestrated_agents.py

Code-Orchestrated Multi-Agent University Task Demo

This module demonstrates advanced multi-agent orchestration patterns for university-related tasks using the OpenAI Agents SDK. Features:
- Structured output models for task classification, study planning, evaluation
- Specialized agents for classification, planning, summary writing, MCQ generation, and evaluation
- Four orchestration patterns: structured routing, sequential chaining, parallel execution, iterative improvement
- Robust error handling and visually distinct, educational output
- Environment variable: GEMINI_API_KEY (required)

Author: Zohaib Khan
References: OpenAI Agents SDK documentation
"""

import asyncio
from enum import Enum
import os
from typing import List
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
    AgentOutputSchema,
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
# Structured Output Models
# =========================

class TaskCategory(str, Enum):
    """Enumeration of possible university task categories."""
    STUDY_PLANNING = "study_planning"
    SUMMARY_WRITING = "summary_writing"
    MCQ_GENERATION = "mcq_generation"
    UNKNOWN = "unknown"

class TaskClassification(BaseModel):
    """Model for classifying a university-related task."""
    category: TaskCategory
    urgency: str  # "low", "medium", "high"
    subject: str
    reasoning: str

class StudyPlanStructure(BaseModel):
    """Model for a structured study plan."""
    subject: str
    chapters: List[str]
    daily_hours: int
    duration_days: int
    plan_notes: str

class EvaluationReport(BaseModel):
    """Model for evaluating academic responses."""
    overall_score: int  # 1 to 10
    comments: str
    suggestions: List[str]
    passes_threshold: bool

# =========================
# Specialized Agents
# =========================

task_classifier = Agent(
    name="TaskClassifierAgent",
    instructions="""
    You are a task classification expert. Decide what kind of university-related task the user is giving you.

    Categories:
    - STUDY_PLANNING: tasks like 'plan my week for data science'
    - SUMMARY_WRITING: tasks like 'summarize lecture 3 of calculus'
    - MCQ_GENERATION: tasks like 'create MCQs for OS chapter 5'

    Also mention urgency and subject, and explain why you made this decision.
    """,
    output_type=TaskClassification,
    model=model,
)

study_planner = Agent(
    name="StudyPlannerAgent",
    instructions="""
    You are a study planning assistant. Given a subject and chapters, make a plan for how many hours per day the student should study and how many days it will take.

    Give clear notes to help student follow the plan.
    """,
    output_type=StudyPlanStructure,
    model=model,
)

summary_writer = Agent(
    name="SummaryWriterAgent",
    instructions="""
    You are a summary writer for university students. Given a lecture or topic, write a clear and simple summary that helps them revise.
    """,
)

mcq_generator = Agent(
    name="MCQGeneratorAgent",
    instructions="""
    You generate 5 good multiple choice questions based on the topic provided. Keep each question with 4 options and mark the correct one.
    """,
    model=model,
)

quality_evaluator = Agent(
    name="QualityEvaluatorAgent",
    instructions="""
    You evaluate the quality of academic responses (like summaries or MCQs). Score from 1 to 10 and explain what was good or needs improvement.

    Mention if the work passes minimum threshold of 7.
    """,
    output_type=EvaluationReport,
    model=model,
)

# =========================
# Orchestration Patterns
# =========================

async def pattern_structured_routing_university():
    """Pattern 1: Structured Routing in University Domain.
    Classifies user tasks and routes to the appropriate agent.
    """
    print("\n==============================")
    print("📚 [PATTERN 1] University Task Classifier & Router")
    print("==============================\n")
    # Simulated user tasks
    tasks = [
        "Make a 5-day study plan for Data Structures chapters 1-5",
        "Summarize lecture 3 of Linear Algebra",
        "Create 5 MCQs for Operating Systems Unit 4",
        "Plan my week for Algorithms and Theory of Computation",
    ]
    results = []
    for task in tasks:
        print(f"\n📥 [INPUT] Task: {task}")
        # Step 1: Classify the task
        classification_result = await Runner.run(task_classifier, task)
        classification = classification_result.final_output_as(TaskClassification)
        print(f"🔍 [DEBUG] Category: {classification.category}")
        print(f"🧠 [DEBUG] Subject: {classification.subject}")
        print(f"⚡ [DEBUG] Urgency: {classification.urgency}")
        print(f"💬 [DEBUG] Reason: {classification.reasoning}")
        # Step 2: Route to appropriate agent
        if classification.category == TaskCategory.STUDY_PLANNING:
            agent = study_planner
        elif classification.category == TaskCategory.SUMMARY_WRITING:
            agent = summary_writer
        elif classification.category == TaskCategory.MCQ_GENERATION:
            agent = mcq_generator
        else:
            print("⚠️ [WARNING] Unknown category. Skipping.")
            continue
        # Step 3: Run selected agent
        result = await Runner.run(agent, task)
        print(f"✅ [RESULT] {agent.name} completed the task.")
        print(f"📤 [OUTPUT]:\n{result.final_output}\n")
        results.append(
            {
                "task": task,
                "classified_as": classification.category,
                "agent_used": agent.name,
                "output": result.final_output,
            }
        )
    return results

async def pattern_sequential_chaining_university():
    """Pattern 2: Sequential Agent Chaining for Study Summary Plan.
    Generates a study plan, then writes and evaluates summaries for each chapter.
    """
    print("\n==============================")
    print("🔗 [PATTERN 2] Sequential Chaining – Study Summary Plan")
    print("==============================\n")
    subject = "Machine Learning"
    user_request = f"Make a 5-day study plan and summary for {subject} chapters 1 to 5"
    # Step 1: Create Study Plan
    print("📘 [STEP 1] Generating study plan...")
    planner_result = await Runner.run(study_planner, user_request)
    plan = planner_result.final_output_as(StudyPlanStructure)
    print(f"📚 [DEBUG] Subject: {plan.subject}")
    print(f"📆 [DEBUG] Duration: {plan.duration_days} days, {plan.daily_hours} hrs/day")
    print(f"🗂️ [DEBUG] Chapters: {plan.chapters}")
    all_results = []
    # Step 2: For each chapter, write a summary
    for chapter in plan.chapters:
        print(f"\n✍️ [STEP 2] Writing summary for {chapter}")
        summary_prompt = f"Write a detailed summary for {plan.subject} - {chapter}"
        summary_result = await Runner.run(summary_writer, summary_prompt)
        summary_text = summary_result.final_output
        # Step 3: Evaluate the summary
        print("🧪 [STEP 3] Evaluating summary quality...")
        eval_prompt = f"Evaluate this summary for clarity and usefulness:\n\n{summary_text}"
        eval_result = await Runner.run(quality_evaluator, eval_prompt)
        report = eval_result.final_output_as(EvaluationReport)
        print(f"✅ [RESULT] Summary Score: {report.overall_score}/10")
        print(f"📝 [DEBUG] Suggestions: {report.suggestions}")
        print(f"🎯 [DEBUG] Pass: {report.passes_threshold}")
        all_results.append({
            "chapter": chapter,
            "summary": summary_text,
            "evaluation": report
        })
    return {
        "study_plan": plan,
        "summaries": all_results
    }

async def pattern_parallel_execution_university():
    """Pattern 3: Parallel Summary Writing for Multiple Subjects.
    Runs summary writing and evaluation in parallel for multiple topics.
    """
    print("\n==============================")
    print("🚀 [PATTERN 3] Parallel Execution – Summarize Topics in Parallel")
    print("==============================\n")
    topics = [
        "Unit 1 of DBMS",
        "Chapter 3 of Operating Systems",
        "Lecture 5 of Computer Networks",
        "Module 2 of AI",
        "Lecture 7 of Software Engineering"
    ]
    async def summarize_and_evaluate(topic: str):
        print(f"✍️ [DEBUG] Writing summary for: {topic}")
        summary_result = await Runner.run(summary_writer, f"Summarize: {topic}")
        summary = summary_result.final_output
        print(f"🧪 [DEBUG] Evaluating summary for: {topic}")
        eval_result = await Runner.run(quality_evaluator, f"Evaluate this summary:\n\n{summary}")
        report = eval_result.final_output
        return {
            "topic": topic,
            "summary": summary,
            "evaluation": report
        }
    print(f"⚡ [INFO] Running {len(topics)} summaries in parallel...")
    start_time = asyncio.get_event_loop().time()
    results = await asyncio.gather(*[
        summarize_and_evaluate(topic) for topic in topics
    ])
    end_time = asyncio.get_event_loop().time()
    print(f"✅ [INFO] Parallel execution finished in {end_time - start_time:.2f} seconds")
    for r in results:
        print(f"\n📝 [OUTPUT] {r['topic']}")
        print(f"Summary:\n{r['summary']}")
        print(f"Evaluation:\n{r['evaluation']}")
    return {
        "topics": [r['topic'] for r in results],
        "summaries": results,
        "execution_time": end_time - start_time
    }

async def pattern_iterative_improvement_university():
    """Pattern 4: Iterative Improvement Loop for Summary Quality.
    Iteratively improves a summary based on evaluation feedback until a score threshold is met.
    """
    print("\n==============================")
    print("♻️ [PATTERN 4] Iterative Summary Improvement")
    print("==============================\n")
    topic = "What is Normalization in DBMS?"
    max_iterations = 3
    score_threshold = 8
    current_summary = None
    last_evaluation: EvaluationReport = None
    for i in range(1, max_iterations + 1):
        print(f"\n🔁 [DEBUG] Iteration {i}/{max_iterations}")
        if current_summary is None:
            # First draft
            print("✍️ [DEBUG] Creating initial summary...")
            result = await Runner.run(summary_writer, f"Write a summary on: {topic}")
            current_summary = result.final_output
        else:
            # Improvement based on last feedback
            print("🔧 [DEBUG] Improving summary based on feedback...")
            improvement_prompt = f"""
            Improve the following summary based on this feedback:

            FEEDBACK:
            Score: {last_evaluation.overall_score}
            Suggestions: {', '.join(last_evaluation.suggestions)}

            CURRENT SUMMARY:
            {current_summary}

            TOPIC:
            {topic}
            """
            result = await Runner.run(summary_writer, improvement_prompt)
            current_summary = result.final_output
        # Evaluation
        print("🧪 [DEBUG] Evaluating summary...")
        eval_result = await Runner.run(quality_evaluator, f"Evaluate this summary:\n\n{current_summary}")
        last_evaluation = eval_result.final_output_as(EvaluationReport)
        print(f"📊 [DEBUG] Score: {last_evaluation.overall_score}/10")
        print(f"📝 [DEBUG] Suggestions: {last_evaluation.suggestions}")
        print(f"🎯 [DEBUG] Pass: {last_evaluation.passes_threshold}")
        if last_evaluation.overall_score >= score_threshold:
            print("✅ [RESULT] Quality threshold met!")
            break
    return {
        "final_summary": current_summary,
        "final_score": last_evaluation.overall_score,
        "iterations": i,
        "final_feedback": last_evaluation
    }

# =========================
# Main Entrypoint
# =========================

if __name__ == "__main__":
    print("\n==============================")
    print("🎓 [MAIN] University Multi-Agent Orchestration Patterns Demo 🎓")
    print("==============================\n")
    # You can run any pattern here for demonstration:
    # asyncio.run(pattern_structured_routing_university())
    # asyncio.run(pattern_sequential_chaining_university())
    # asyncio.run(pattern_parallel_execution_university())
    asyncio.run(pattern_iterative_improvement_university())
    print("\n✅ [INFO] All demos completed.\n")
