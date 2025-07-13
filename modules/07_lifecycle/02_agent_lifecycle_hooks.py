import asyncio
import time
from datetime import datetime
from typing import Any, Dict, Optional, List

from agents import (
    Agent,
    function_tool,
    Runner,
    RunContextWrapper,
    AgentHooks,
    OpenAIChatCompletionsModel,
    set_trace_processors,
    set_tracing_disabled,
)
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
set_tracing_disabled(True)

# Dummy Gemini key setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.0-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# External model setup
client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
model = OpenAIChatCompletionsModel(openai_client=client, model=GEMINI_MODEL_NAME)

# ================================
# 1. Tools for Student Support Agent
# ================================


@function_tool
def faq_lookup(question: str) -> str:
    """Look up answers from student FAQs."""
    return f"Found answer to your question: '{question}' → Please check the university handbook."


@function_tool
def submit_complaint(issue: str) -> str:
    """Submit a complaint ticket."""
    return (
        f"Complaint submitted for: '{issue}'. Your ticket ID is ST-{hash(issue) % 1000}"
    )

@function_tool
def search_docs(query: str) -> str:
    """Search documentation."""
    return f"🔍 Search results for '{query}': [Doc1, Doc2, Doc3]"

@function_tool
def log_issue(description: str) -> str:
    """Log a customer issue."""
    return f"📝 Issue logged: {description} (Issue ID: ISS-{hash(description) % 10000})"

@function_tool
def escalate_to_expert(reason: str) -> str:
    """Escalate the issue to a senior expert."""
    return f"⚠️ Issue escalated to expert team. Reason: {reason}"

# ================================
# 2. Basic AgentHooks (Beginner)
# ================================


class BeginnerAgentHooks(AgentHooks):
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.activation_count = 0
        self.tool_usage_count = 0
        self.start_time: Optional[float] = None
        self.processing_times: List[float] = []

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.activation_count += 1
        self.start_time = time.time()
        print(f"\n🎬 [{agent.name}] Starting Run #{self.activation_count}")

    async def on_end(self, context: Any, agent: Agent, output: Any) -> None:
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.processing_times.append(elapsed)
            print(f"✅ [{agent.name}] Finished Run in {elapsed:.2f}s")

    async def on_tool_start(self, context: Any, agent: Agent, tool) -> None:
        self.tool_usage_count += 1
        print(f"🔧 [{agent.name}] Started using tool: {tool.name}")

    async def on_tool_end(self, context: Any, agent: Agent, tool, result: str) -> None:
        print(f"✅ [{agent.name}] Finished using tool: {tool.name}")

    def print_summary(self):
        print("\n📊 Agent Summary:")
        print(f"Agent Name: {self.agent_name}")
        print(f"Total Runs: {self.activation_count}")
        print(f"Total Tool Uses: {self.tool_usage_count}")
        if self.processing_times:
            print(
                f"Avg Processing Time: {sum(self.processing_times)/len(self.processing_times):.2f}s"
            )


class IntermediateAgentHooks(AgentHooks):
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.session_count = 0
        self.total_processing_time = 0.0
        self.sessions: List[Dict[str, Any]] = []
        self.active_sessions: Dict[str, float] = {}  # session_id → start_time

    async def on_start(self, context: Any, agent: Agent) -> None:
        self.session_count += 1
        session_id = f"{agent.name}-session-{self.session_count}"
        self.active_sessions[session_id] = time.time()

        # Save minimal session info
        session = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "input_preview": str(getattr(context, "input", ""))[:80],
        }
        self.sessions.append(session)

        print(f"🎬 [{agent.name}] Session started → ID: {session_id}")

    async def on_end(self, context: Any, agent: Agent, output: Any) -> None:
        session_id = f"{agent.name}-session-{self.session_count}"
        start_time = self.active_sessions.pop(session_id, None)

        if start_time is None:
            print(f"⚠️ No start time found for session: {session_id}")
            return

        duration = time.time() - start_time
        self.total_processing_time += duration

        # Find session and update
        for session in self.sessions:
            if session["session_id"] == session_id:
                session.update({
                    "duration": duration,
                    "output_preview": str(output)[:80],
                })
                break

        print(f"✅ [{agent.name}] Session {session_id} ended → Duration: {duration:.2f}s")

    async def on_tool_start(self, context: Any, agent: Agent, tool) -> None:
        session_id = f"{agent.name}-session-{self.session_count}"
        for session in self.sessions:
            if session["session_id"] == session_id:
                session.setdefault("tools_used", []).append({
                    "tool_name": tool.name,
                    "start": time.time(),
                })
        print(f"🔧 [{agent.name}] Tool started: {tool.name}")

    async def on_tool_end(self, context: Any, agent: Agent, tool, result: str) -> None:
        session_id = f"{agent.name}-session-{self.session_count}"
        for session in self.sessions:
            if session["session_id"] == session_id:
                for t in reversed(session.get("tools_used", [])):
                    if t["tool_name"] == tool.name and "end" not in t:
                        t["end"] = time.time()
                        t["duration"] = t["end"] - t["start"]
                        t["result_preview"] = str(result)[:60]
                        break
        print(f"✅ [{agent.name}] Tool ended: {tool.name}")

    def report(self):
        print(f"\n📊 Intermediate Report for {self.agent_name}")
        print(f"Total Sessions: {self.session_count}")
        if self.session_count:
            print(f"Average Duration: {self.total_processing_time / self.session_count:.2f}s")
        for session in self.sessions:
            print(f"  • {session['session_id']} ({session.get('duration', 0):.2f}s)")
            print(f"    ↳ Input: {session.get('input_preview', '')}")
            print(f"    ↳ Output: {session.get('output_preview', '')}")
            for t in session.get("tools_used", []):
                print(f"      🔧 {t['tool_name']} ({t.get('duration', 0):.2f}s): {t.get('result_preview', '')}")
                

class AdvancedAgentHooks(AgentHooks):
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.session_id_counter = 0
        self.sessions: List[Dict[str, Any]] = []
        self.active_sessions: List[tuple[str, float]] = []
        self.response_scores: List[float] = []
        self.optimization_notes: List[str] = []

    async def on_start(self, context: Any, agent: Agent) -> None:
        self.session_id_counter += 1
        session_id = f"{agent.name}-session-{self.session_id_counter}"
        start_time = time.time()

        session = {
            "session_id": session_id,
            "start_time": start_time,
            "input": str(getattr(context, "input", ""))[:150],
            "tools_used": []
        }

        self.sessions.append(session)
        self.active_sessions.append((session_id, start_time))

        print(f"🚀 [{agent.name}] Started Session {session_id}")

    async def on_end(self, context: Any, agent: Agent, output: Any) -> None:
        session_id, start_time = self.active_sessions.pop()

        duration = time.time() - start_time
        session = self._get_session_by_id(session_id)
        session["end_time"] = time.time()
        session["duration"] = duration
        session["output"] = str(output)[:200]

        score = self._score_response(str(output), duration)
        self.response_scores.append(score)

        print(f"✅ [{agent.name}] Ended {session_id} | Duration: {duration:.2f}s | Quality Score: {score:.1f}")

        if duration > 4.0:
            print(f"⚠️  Slow response detected for {session_id}")

        if score < 60:
            suggestion = f"Low quality output in {session_id}, needs better response structuring."
            if suggestion not in self.optimization_notes:
                self.optimization_notes.append(suggestion)

    async def on_tool_start(self, context: Any, agent: Agent, tool) -> None:
        session_id = f"{agent.name}-session-{self.session_id_counter}"
        session = self._get_session_by_id(session_id)

        session["tools_used"].append({
            "tool_name": tool.name,
            "start": time.time(),
        })

        print(f"🔧 [{agent.name}] Starting tool {tool.name}")

    async def on_tool_end(self, context: Any, agent: Agent, tool, result: str) -> None:
        session_id = f"{agent.name}-session-{self.session_id_counter}"
        session = self._get_session_by_id(session_id)

        for t in reversed(session["tools_used"]):
            if t["tool_name"] == tool.name and "end" not in t:
                t["end"] = time.time()
                t["duration"] = t["end"] - t["start"]
                t["result"] = str(result)[:100]
                break

        print(f"✅ [{agent.name}] Finished tool {tool.name}")

    async def on_handoff(self, context: Any, agent: Agent, source: Agent) -> None:
        print(f"🔄 [{agent.name}] Received handoff from {source.name}")
        session_id = f"{agent.name}-session-{self.session_id_counter}"
        session = self._get_session_by_id(session_id)
        session["handoff_from"] = source.name

    def _get_session_by_id(self, session_id: str) -> Dict[str, Any]:
        for session in self.sessions:
            if session["session_id"] == session_id:
                return session
        return {}

    def _score_response(self, output: str, duration: float) -> float:
        score = 50
        if len(output) > 80:
            score += 20
        if "thank" in output.lower() or "please" in output.lower():
            score += 10
        if duration < 2.0:
            score += 10
        elif duration < 5.0:
            score += 5
        else:
            score -= 10
        return max(0, min(score, 100))

    def get_full_report(self) -> Dict[str, Any]:
        if not self.sessions:
            return {
                "status": "No sessions recorded",
                "agent": self.agent_name,
            }

        durations = [s["duration"] for s in self.sessions if "duration" in s]
        total_tool_uses = sum(len(s.get("tools_used", [])) for s in self.sessions)

        return {
            "agent": self.agent_name,
            "summary": {
                "total_sessions": len(self.sessions),
                "total_tool_uses": total_tool_uses,
                "average_duration": sum(durations) / len(durations) if durations else 0,
                "fastest_response": min(durations) if durations else None,
                "slowest_response": max(durations) if durations else None,
            },
            "recent_sessions": self.sessions[-3:],
            "response_scores": self.response_scores[-3:],
            "optimization_notes": self.optimization_notes[-3:],
        }



# ================================
# 3. Create Agent
# ================================


def create_student_support_agent() -> Agent:
    agent = Agent(
        name="StudentSupportAgent",
        instructions="You help university students with common issues and questions.",
        tools=[faq_lookup, submit_complaint],
        model=model,
    )
    agent.hooks = BeginnerAgentHooks("StudentSupportAgent")
    return agent


def create_intermediate_agent() -> Agent:
    agent = Agent(
        name="StudentSupportIntermediateAgent",
        instructions="You help students with policy and complaints. You track sessions and tool usage.",
        tools=[faq_lookup, submit_complaint],
        model=model,
    )
    agent.hooks = IntermediateAgentHooks("StudentSupportIntermediateAgent")
    return agent

def create_advanced_monitored_agent() -> Agent:
    """Create an agent with advanced lifecycle monitoring."""
    agent = Agent(
        name="AdvancedSupportAgent",
        instructions="You are a highly observant support agent. Track and improve your performance across sessions.",
        tools=[search_docs, log_issue, escalate_to_expert],
        model=model,
    )

    # Attach advanced hooks
    agent.hooks = AdvancedAgentHooks(agent_name="AdvancedSupportAgent")
    return agent


# ================================
# 4. Run Demo
# ================================


async def demo_basic_agent_hooks():
    print("\n🎓 Demo: Beginner AgentHooks Example")

    agent = create_student_support_agent()

    inputs = [
        "How can I apply for a scholarship?",
        "I want to report a hostel issue.That's a complain. There is no light",
    ]

    for user_input in inputs:
        result = await Runner.run(agent, input=user_input)
        print(f"🗣️ User Input: {user_input}")
        print(f"🤖 Agent Output: {result.final_output}\n")

    # Print simple summary
    if isinstance(agent.hooks, BeginnerAgentHooks):
        agent.hooks.print_summary()

async def demo_intermediate_agent_hooks():
    print("\n=== Intermediate Agent Hooks Demo ===")
    agent = create_intermediate_agent()

    queries = [
        "How to request a leave certificate?",
        "My hostel room has plumbing issues.",
    ]

    for query in queries:
        result = await Runner.run(agent, input=query)
        print(f"\n🗣️ Input: {query}")
        print(f"🤖 Output: {result.final_output}")

    if isinstance(agent.hooks, IntermediateAgentHooks):
        agent.hooks.report()

async def run_advanced_hooks_demo():
    print("🔍 Starting Advanced AgentHooks Demo")

    agent = create_advanced_monitored_agent()

    # Simulated user interactions
    interactions = [
        "How do I reset my password?",
        "This is urgent. My account is locked.",
        "Can you escalate this to a supervisor?",
        "Where can I find help with billing issues?",
        "System is crashing again and again.",
    ]

    for idx, msg in enumerate(interactions, 1):
        print(f"\n--- Interaction #{idx} ---")
        result = await Runner.run(agent, input=msg)
        print(f"🧠 Output: {result.final_output}")

    # Show final performance report
    if isinstance(agent.hooks, AdvancedAgentHooks):
        report = agent.hooks.get_full_report()
        for key, val in report.items():
            print(f"\n🗂 {key}:")
            if isinstance(val, dict):
                for k, v in val.items():
                    print(f"  {k}: {v}")
            elif isinstance(val, list):
                for item in val:
                    print(f"  - {item}")
            else:
                print(f"  {val}")


# ================================
# 5. Run Main
# ================================
async def main():
    print("🎓 Running AgentHooks Demonstrations")
    print("=" * 60)

    # await demo_basic_agent_hooks()
    # await demo_intermediate_agent_hooks()
    await run_advanced_hooks_demo()


if __name__ == "__main__":
    asyncio.run(main())
