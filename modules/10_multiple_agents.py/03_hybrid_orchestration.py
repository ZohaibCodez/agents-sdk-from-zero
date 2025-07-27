"""
modules/10_multiple_agents.py/03_hybrid_orchestration.py

Hybrid Orchestration Multi-Agent University Task Demo

This module demonstrates adaptive hybrid orchestration patterns for university-related tasks using the OpenAI Agents SDK. Features:
- Dynamic strategy selection (LLM-driven, code-driven, hybrid)
- Specialized agents for planning, notes, motivation, tools, and execution
- Robust fallback and error handling
- Visually distinct, educational output for all orchestration modes
- Environment variable: MISTRAL_API_KEY (required)

Author: Zohaib Khan
References: OpenAI Agents SDK documentation
"""

import asyncio
from enum import Enum
import os
from typing import List, Literal
from agents.extensions.models.litellm_model import LitellmModel
from pydantic import BaseModel
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    handoff,
)
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI

# =========================
# Environment & Model Setup
# =========================
load_dotenv(find_dotenv())

MISTRAL_API_KEY: str | None = os.getenv("MISTRAL_API_KEY")
MODEL_NAME: str = "mistral/mistral-small-2506"

if not MISTRAL_API_KEY:
    print("\n❌ [ERROR] MISTRAL_API_KEY environment variable is required but not found.\n")
    raise ValueError("MISTRAL_API_KEY environment variable is required but not found.")

print(f"\n==============================")
print(f"🚀 Initializing OpenRouter client with model: {MODEL_NAME}")
print(f"==============================\n")
model = LitellmModel(model=MODEL_NAME, api_key=MISTRAL_API_KEY)
print(f"✅ [INFO] Model configured successfully\n")

# =========================
# Structured Output Models
# =========================

class OrchestrationMode(str, Enum):
    """Enumeration of orchestration strategies. Author: Zohaib Khan"""
    LLM = "llm_driven"
    CODE = "code_driven"
    HYBRID = "hybrid"

class TaskComplexity(str, Enum):
    """Enumeration of task complexity levels. Author: Zohaib Khan"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class StudyTaskStrategy(BaseModel):
    """Model for planning orchestration strategy for a study task. Author: Zohaib Khan"""
    complexity: TaskComplexity
    domains: List[str]  # ["study_plan", "notes", "tools", "motivation"]
    creativity_required: bool
    orchestration_mode: OrchestrationMode
    fallback_mode: OrchestrationMode
    reasoning: str

class StudyTask(BaseModel):
    """Model for a study task input. Author: Zohaib Khan"""
    task_id: str
    topic: str
    goal: str
    task_type: Literal["motivation", "plan", "tools", "notes", "undefined"]

# =========================
# Agent Definitions
# =========================

study_task_planner = Agent(
    name="StudyTaskPlanner",
    instructions="""
    You are a university task strategy planner. Analyze the given task and decide:

    - Complexity: easy / medium / hard
    - Domains involved: study_plan, motivation, notes, tools
    - Whether it requires creative thinking
    - Orchestration strategy: llm_driven, code_driven, hybrid
    - A fallback strategy (different from the main)
    - Clear reasoning for your choices

    Guidelines:
    - Use LLM for creative, motivational, or ambiguous tasks
    - Use CODE for structured, systematic tasks with clear steps
    - Use HYBRID for complex tasks needing both creativity and structure

    Return a structured strategy accordingly.
    """,
    output_type=StudyTaskStrategy,
    model=model,
)

study_planner_agent = Agent(
    name="StudyPlannerAgent",
    instructions="""
    You are a study planning assistant for university students.
    Your job is to:

    - Break down learning goals into weekly plans
    - Allocate study time realistically
    - Recommend focus areas based on goals
    - Include breaks and review sessions
    - Provide actionable schedules

    Be practical and considerate of student workload.
    """,
    model=model,
)

notes_generator_agent = Agent(
    name="NotesGeneratorAgent",
    instructions="""
    You generate structured notes for students based on topics.

    Format notes with:
    - Clear headings and subheadings
    - Bullet points for key concepts
    - Examples where helpful
    - Summary sections
    - Action items or key takeaways

    Make notes scannable and study-friendly.
    """,
    model=model,
)

motivation_coach_agent = Agent(
    name="MotivationCoachAgent",
    instructions="""
    You are a motivational coach for overwhelmed students.

    Provide:
    - Encouraging and empathetic words
    - Practical productivity hacks
    - Mindset tips to stay focused
    - Stress management techniques
    - Realistic goal-setting advice

    Be supportive but also practical in your advice.
    """,
    model=model,
)

toolchain_expert_agent = Agent(
    name="ToolChainExpertAgent",
    instructions="""
    You recommend tools to help students learn and build faster.

    Suggest:
    - AI tools like LangGraph, Agents SDK, Docker
    - Learning platforms and resources
    - Study apps and productivity tools
    - Best practices for tool integration
    - Cost-effective solutions for students

    Focus on tools that genuinely help productivity.
    """,
    model=model,
)

executor_agent = Agent(
    name="ExecutorAgent",
    instructions="""
    You execute deterministic, structured tasks with systematic thinking.

    Your approach:
    - Break down complex tasks into clear steps
    - Use standard formatting and structure
    - Provide detailed explanations
    - Follow logical sequences
    - Ensure completeness and accuracy

    Focus on clarity and systematic execution.
    """,
    model=model,
)

# =========================
# Hybrid Orchestrator
# =========================

class MyHybridOrchestrator:
    """Hybrid orchestrator for adaptive multi-agent task execution. Author: Zohaib Khan"""
    def __init__(self):
        self.domain_agents = {
            "study_plan": study_planner_agent,
            "notes": notes_generator_agent,
            "motivation": motivation_coach_agent,
            "tools": toolchain_expert_agent,
        }

    async def analyze_task(self, task_input) -> StudyTaskStrategy:
        """Analyze task to determine orchestration strategy. Author: Zohaib Khan"""
        print("\n🔍 [INFO] Analyzing task strategy...")
        # Handle both string and StudyTask object inputs
        if isinstance(task_input, StudyTask):
            analysis_prompt = f"""
            Task: {task_input.topic}
            Goal: {task_input.goal}
            Type: {task_input.task_type}
            
            Analyze this task and provide orchestration strategy.
            """
        else:
            analysis_prompt = str(task_input)
        print("[DEBUG] Prompt generated for strategy analysis.")
        try:
            result = await Runner.run(study_task_planner, analysis_prompt)
        except Exception as e:
            print(f"❌ [ERROR] Strategy analysis failed: {e}")
            raise
        print(f"[OUTPUT] Agent response: {result.final_output}")
        strategy = result.final_output_as(StudyTaskStrategy)
        print(f"   📊 [DEBUG] Complexity: {strategy.complexity.value}")
        print(f"   🎯 [DEBUG] Mode: {strategy.orchestration_mode.value}")
        print(f"   🔄 [DEBUG] Fallback: {strategy.fallback_mode.value}")
        print(f"   💡 [DEBUG] Reasoning: {strategy.reasoning}")
        return strategy

    async def execute_llm_driven(self, task_input, strategy: StudyTaskStrategy) -> dict:
        """Execute using LLM-driven orchestration (agents work autonomously). Author: Zohaib Khan"""
        print("\n🤖 [INFO] Executing with LLM-driven strategy...")
        task_str = self._get_task_string(task_input)
        involved_agents = [self.domain_agents[d] for d in strategy.domains if d in self.domain_agents]
        execution_summary = []
        # Configure handoffs for collaborative work
        if len(involved_agents) > 1:
            main_agent = involved_agents[0]
            other_agents = involved_agents[1:]
            main_agent.handoffs = [handoff(agent=agent) for agent in other_agents]
            print(f"   🎭 [DEBUG] Main agent: {main_agent.name} with {len(other_agents)} collaborators")
            enhanced_prompt = f"""
            Task: {task_str}
            
            You can collaborate with these domain experts as needed:
            {[agent.name for agent in other_agents]}
            
            Use your autonomy to:
            1. Plan your approach
            2. Delegate to specialists when helpful
            3. Synthesize results creatively
            
            Deliver a comprehensive solution.
            """
            result = await Runner.run(main_agent, enhanced_prompt, max_turns=5)
            execution_summary.append((main_agent.name, result.final_output))
        else:
            agent = involved_agents[0]
            print(f"   🎯 [DEBUG] Single agent: {agent.name}")
            result = await Runner.run(agent, task_str)
            execution_summary.append((agent.name, result.final_output))
        return {
            "mode": "llm_driven",
            "agents_used": [agent.name for agent in involved_agents],
            "results": execution_summary,
            "final_output": (
                execution_summary[-1][1] if execution_summary else "No output generated"
            ),
        }

    async def execute_code_driven(self, task_input, strategy: StudyTaskStrategy) -> dict:
        """Execute using code-driven orchestration (systematic step-by-step). Author: Zohaib Khan"""
        print("\n⚙️ [INFO] Executing with code-driven strategy...")
        task_str = self._get_task_string(task_input)
        step_outputs = []
        # Step 1: Task decomposition
        print("   📋 [DEBUG] Step 1: Task decomposition...")
        breakdown_prompt = f"Break down this task into clear, actionable steps: {task_str}"
        step1 = await Runner.run(executor_agent, breakdown_prompt)
        step_outputs.append(("task_decomposition", step1.final_output))
        # Step 2: Domain expert execution (sequential)
        for i, domain in enumerate(strategy.domains, 2):
            if domain in self.domain_agents:
                print(f"   🎯 [DEBUG] Step {i}: {domain} domain execution...")
                domain_prompt = f"""
                Original task: {task_str}
                Task breakdown: {step1.final_output}
                
                Execute your {domain} specialization for this task.
                """
                result = await Runner.run(self.domain_agents[domain], domain_prompt)
                step_outputs.append((f"{domain}_execution", result.final_output))
        # Step 3: Final synthesis
        print(f"   🔄 [DEBUG] Step {len(step_outputs) + 1}: Final synthesis...")
        synthesis_prompt = f"""
        Original task: {task_str}
        
        Synthesis all execution results:
        {chr(10).join([f"- {step[0]}: {step[1][:200]}..." for step in step_outputs])}
        
        Provide final, complete, and polished result.
        """
        final_result = await Runner.run(executor_agent, synthesis_prompt)
        return {
            "mode": "code_driven",
            "total_steps": len(step_outputs) + 1,
            "steps": step_outputs,
            "final_output": final_result.final_output,
        }

    async def execute_hybrid(self, task_input, strategy: StudyTaskStrategy) -> dict:
        """Execute using hybrid orchestration (combines creativity with structure). Author: Zohaib Khan"""
        print("\n🔀 [INFO] Executing with hybrid strategy...")
        task_str = self._get_task_string(task_input)
        phases = []
        # Phase 1: Creative strategic planning (LLM-driven)
        print("   🎨 [DEBUG] Phase 1: Creative strategic planning...")
        strategy_prompt = f"""
        Develop a creative and innovative strategy for: {task_str}
        
        Consider:
        - Unique approaches and possibilities
        - Strategic framework and vision  
        - Key insights and opportunities
        - High-level action plan
        
        Focus on creative thinking and strategic insight.
        """
        strategy_phase = await Runner.run(motivation_coach_agent, strategy_prompt)
        phases.append(("creative_strategy", strategy_phase.final_output))
        # Phase 2: Structured execution planning (Code-driven)
        print("   📊 [DEBUG] Phase 2: Structured execution planning...")
        execution_prompt = f"""
        Create detailed execution plan based on this creative strategy:
        
        STRATEGY: {strategy_phase.final_output}
        ORIGINAL TASK: {task_str}
        
        Provide:
        - Specific action steps
        - Timeline and priorities
        - Resource requirements
        - Success metrics
        """
        execution_plan = await Runner.run(executor_agent, execution_prompt)
        phases.append(("structured_plan", execution_plan.final_output))
        # Phase 3: Domain expert implementation (Code-driven)
        print("   🎯 [DEBUG] Phase 3: Domain expert implementation...")
        domain_results = []
        for domain in strategy.domains:
            if domain in self.domain_agents:
                print(f"     ➤ [DEBUG] {domain} expert...")
                domain_prompt = f"""
                Execute your part of this plan:
                
                CREATIVE STRATEGY: {strategy_phase.final_output}
                EXECUTION PLAN: {execution_plan.final_output}
                ORIGINAL TASK: {task_str}
                
                Focus on {domain}-specific implementation and deliverables.
                """
                result = await Runner.run(self.domain_agents[domain], domain_prompt)
                domain_results.append((domain, result.final_output))
        phases.append(("domain_implementation", domain_results))
        # Phase 4: Creative synthesis and refinement (LLM-driven)
        print("   ✨ [DEBUG] Phase 4: Creative synthesis...")
        final_synthesis_prompt = f"""
        Synthesize all work into final polished deliverable:
        
        ORIGINAL TASK: {task_str}
        CREATIVE STRATEGY: {strategy_phase.final_output}
        EXECUTION PLAN: {execution_plan.final_output}
        DOMAIN RESULTS: {domain_results}
        
        Create final result that:
        - Meets all requirements comprehensively
        - Incorporates creative insights for enhanced impact
        - Provides actionable value to the student
        """
        final_result = await Runner.run(study_planner_agent, final_synthesis_prompt)
        phases.append(("creative_synthesis", final_result.final_output))
        return {
            "mode": "hybrid",
            "total_phases": len(phases),
            "phases": phases,
            "final_output": final_result.final_output,
        }

    def _get_task_string(self, task_input) -> str:
        """Helper to extract task string from various input types. Author: Zohaib Khan"""
        if isinstance(task_input, StudyTask):
            return f"Topic: {task_input.topic}\nGoal: {task_input.goal}\nType: {task_input.task_type}"
        return str(task_input)

    async def orchestrate(self, task_input) -> dict:
        """Main orchestration method with fallback handling. Author: Zohaib Khan"""
        task_str = self._get_task_string(task_input)
        print(f"\n🎯 [INFO] Orchestrating task: {task_str[:100]}...")
        strategy = None  # Initialize strategy variable
        try:
            # Step 1: Analyze task
            strategy = await self.analyze_task(task_input)
            # Step 2: Execute with primary strategy
            if strategy.orchestration_mode == OrchestrationMode.LLM:
                result = await self.execute_llm_driven(task_input, strategy)
            elif strategy.orchestration_mode == OrchestrationMode.CODE:
                result = await self.execute_code_driven(task_input, strategy)
            else:  # HYBRID
                result = await self.execute_hybrid(task_input, strategy)
            result.update(
                {
                    "primary_mode": strategy.orchestration_mode.value,
                    "fallback_used": False,
                    "strategy": strategy.model_dump(),
                    "success": True,
                }
            )
            print(f"✅ [RESULT] Task completed successfully with {result['mode']} orchestration")
            return result
        except Exception as e:
            print(f"❌ [WARNING] Primary orchestration failed: {e}")
            # Check if strategy was successfully created before attempting fallback
            if strategy is None:
                print("❌ [ERROR] Task analysis failed, cannot determine fallback strategy")
                return {
                    "mode": "error",
                    "error": f"Task analysis failed: {str(e)}",
                    "strategy": None,
                    "success": False,
                }
            print(f"🔄 [INFO] Attempting fallback: {strategy.fallback_mode.value}")
            try:
                # Execute fallback strategy
                if strategy.fallback_mode == OrchestrationMode.CODE:
                    fallback_result = await self.execute_code_driven(task_input, strategy)
                elif strategy.fallback_mode == OrchestrationMode.LLM:
                    fallback_result = await self.execute_llm_driven(task_input, strategy)
                else:  # HYBRID fallback
                    fallback_result = await self.execute_hybrid(task_input, strategy)
                fallback_result.update(
                    {
                        "primary_mode": strategy.orchestration_mode.value,
                        "fallback_mode": strategy.fallback_mode.value,
                        "fallback_used": True,
                        "strategy": strategy.model_dump(),
                        "success": True,
                    }
                )
                print(f"✅ [RESULT] Fallback successful with {fallback_result['mode']} orchestration")
                return fallback_result
            except Exception as e2:
                print(f"❌ [ERROR] Fallback also failed: {e2}")
                return {
                    "mode": "error",
                    "error": str(e2),
                    "primary_error": str(e),
                    "strategy": strategy.model_dump() if strategy else None,
                    "success": False,
                }

# =========================
# Demo and Testing
# =========================

async def demo_tasks():
    """Demonstrate different task types and orchestration modes. Author: Zohaib Khan"""
    print("\n🔁 [INFO] Testing Hybrid Orchestration for Study Tasks\n")
    orchestrator = MyHybridOrchestrator()
    test_tasks = [
        # Simple motivation task -> likely LLM-driven
        # "I'm feeling overwhelmed with my computer science coursework and need help getting motivated and finding focus strategies",
        # Structured planning task -> likely CODE-driven
        StudyTask(
            task_id="task2",
            topic="Learn Python async programming and concurrency",
            goal="Master async/await, threading, and multiprocessing in 4 weeks",
            task_type="plan",
        ),
        # # Complex mixed task -> likely HYBRID
        # StudyTask(
        #     task_id="task3",
        #     topic="Build a complete study system using AI tools",
        #     goal="Create automated note-taking, scheduling, and progress tracking",
        #     task_type="tools",
        # ),
        # # Notes generation -> could be any mode
        # StudyTask(
        #     task_id="task4",
        #     topic="Object-oriented programming principles",
        #     goal="Create comprehensive study notes with examples",
        #     task_type="notes",
        # ),
    ]
    results = []
    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'='*60}")
        print(f"🎯 [TEST CASE {i}] Task")
        print(f"{'='*60}")
        result = await orchestrator.orchestrate(task)
        results.append(result)
        if result.get("success"):
            print(f"✅ [RESULT] Completed using {result.get('mode', 'unknown')} orchestration")
            if result.get("fallback_used"):
                print(f"   🔄 [INFO] (Used fallback: {result.get('fallback_mode')})")
            # Show brief output preview
            final_output = result.get("final_output", "")
            preview = (
                final_output[:200] + "..." if len(final_output) > 200 else final_output
            )
            print(f"📄 [OUTPUT PREVIEW]: {preview}")
        else:
            print(f"❌ [ERROR] Failed: {result.get('error', 'Unknown error')}")
        print(f"\n{'-'*40}")
        # Small delay between tasks for readability
        await asyncio.sleep(1)
    return results

async def main():
    """Main demo execution. Author: Zohaib Khan"""
    print("\n==============================")
    print("🎓 [MAIN] Hybrid Multi-Agent Study Assistant Demo 🎓")
    print("==============================\n")
    print("This demonstrates adaptive orchestration for educational tasks\n")
    try:
        results = await demo_tasks()
        print(f"\n{'='*60}")
        print("📊 [EXECUTION SUMMARY]")
        print(f"{'='*60}")
        successful_tasks = [r for r in results if r.get("success")]
        failed_tasks = [r for r in results if not r.get("success")]
        print(f"✅ [INFO] Successful tasks: {len(successful_tasks)}/{len(results)}")
        print(f"❌ [INFO] Failed tasks: {len(failed_tasks)}")
        if successful_tasks:
            mode_counts = {}
            fallback_count = 0
            for result in successful_tasks:
                mode = result.get("mode", "unknown")
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                if result.get("fallback_used"):
                    fallback_count += 1
            print(f"\n🎯 [INFO] Orchestration modes used:")
            for mode, count in mode_counts.items():
                print(f"   • {mode}: {count} tasks")
            print(f"🔄 [INFO] Fallbacks triggered: {fallback_count}")
        print(f"\n🎓 [KEY LEARNING POINTS]:")
        print(f"• Adaptive strategy selection based on task analysis")
        print(f"• LLM autonomy for creative and motivational tasks")
        print(f"• Systematic execution for structured planning")
        print(f"• Hybrid approach for complex multi-faceted tasks")
        print(f"• Robust error handling with fallback mechanisms")
    except Exception as e:
        print(f"❌ [ERROR] Demo execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
