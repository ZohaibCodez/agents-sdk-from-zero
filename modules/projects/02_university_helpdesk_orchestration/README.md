# 🎓 University Smart Helpdesk Orchestrator

## 🎯 Project Overview

A real-world multi-agent system that simulates a university helpdesk environment. This project showcases advanced orchestration techniques with context-aware routing, escalation handling, and dynamic handoffs using the OpenAI Agents SDK and Mistral models.

## ✨ What This Project Demonstrates

### 🧠 **Advanced Concepts Learned:**
- **Contextual Agent Handoffs**: Passing structured context (`HandoffContext`) between agents
- **Smart Orchestration**: Dynamic routing based on student type, urgency, and case history
- **Escalation Logic**: Tracking escalation chains and routing to department leads
- **Realistic Prompt Engineering**: Domain-specific roles and communication protocols
- **Streaming UX**: Real-time transitions between agents and tools

### 🛠️ **Technical Patterns:**
- Handoff filters and `on_handoff` callbacks
- Enum-based routing intelligence
- Context chaining and escalation tracking
- Modular agent creation with dynamic instructions
- Tool-based decision support for agents

## 🏗️ System Architecture

```

University Helpdesk Orchestrator
├── Helpdesk Support Agent
├── Finance Office Agent
├── Academic Advisor Agent
├── IT Helpdesk Agent
├── Admissions Counselor Agent
├── Student Affairs Agent (Final Escalation)
└── IT Lead Agent (Tech Escalation)

````

## 🚀 Features

- 🧭 **Context-Aware Routing**: Agents adapt behavior using `StudentType` and `QueryUrgency`
- 🔁 **Escalation Tracking**: Tracks how many times a case has been passed
- 🧠 **Orchestration Intelligence**: Prevents circular routing and repeats
- 🛎️ **Department Specialization**: Each agent handles domain-specific queries
- 🎭 **Live Streaming Output**: Logs each step of agent and tool activity in real-time

## 🧱 Code Highlights

### Context Model
```python
class HandoffContext(BaseModel):
    student_type: StudentType
    query_urgency: QueryUrgency
    escalation_count: int = 0
    previous_agents: list[str] = []
    resolution_attempts: list[str] = []
    student_feedback: Optional[str] = None
)
````

### Routing and Escalation Logic

```python
handoff(
    agent=finance,
    tool_name_override="route_to_finance",
    on_handoff=on_student_routing,
    input_type=HandoffContext,
)
```

### Smart Callback Debugging

```python
async def on_escalation_tracking(ctx, handoff_data):
    handoff_data.escalation_count += 1
    print(f"🔁 Escalation #{handoff_data.escalation_count}")
```

## 🎮 How to Run

```bash
# From project root
uv run modules/projects/02_university_helpdesk_orchestration/main.py
```

## 📊 Example Output

```
=== University Helpdesk Orchestration ===
🔧 Tool called: lookup_student_type
📊 Tool Output: international
🔧 Tool called: analyze_query_urgency
📊 Tool Output: critical
🤝 Handing off ...
📌 SMART ROUTING:
   Student Type: international
   Query Urgency: critical
   Escalation Count: 0
   Previous Agents: Helpdesk Support
📤 Switching from Helpdesk Support to IT Helpdesk
🔄 Agent switched to: IT Helpdesk
💬 IT Helpdesk says: I’ve created a ticket and escalated to the IT Lead.
```

## 🧠 Learning Insights

### 💡 Key Breakthroughs

* Mastered the `handoff()` mechanism with `input_type`
* Built a reusable context model with typed enums
* Practiced agent prompt engineering across different roles
* Implemented smart logic for agent-to-agent transitions

### 🔍 Challenges Overcome

* Avoiding redundant routing using `previous_agents`
* Handling urgency and student type in routing logic
* Designing realistic workflows from scratch
* Managing streaming output and tool invocation logs

### 🎯 Real-World Applications

* University support systems
* Call center automation
* Enterprise support routing
* Escalation and triage management bots

## 🔧 Technical Details

* **Language**: Python 3.11+
* **Framework**: OpenAI Agents SDK + LiteLLM
* **Model**: Mistral via OpenRouter
* **Architecture**: Multi-agent with contextual handoffs

## 🎨 Future Enhancements

* [ ] Add persistent context history storage
* [ ] Visual dashboard for routing paths and escalation chains
* [ ] Add feedback-based agent selection
* [ ] Extend to voice agent interface

## 📚 What I Learned

This project helped me deeply understand:

1. **Multi-Agent Workflows**: Chaining and coordinating specialized agents
2. **Context Design**: Modeling shared knowledge across handoffs
3. **Orchestration Intelligence**: Routing based on context and policy
4. **Professional Agent Design**: Writing domain-specific agent roles and behaviors

## 🔗 Related Learning Modules

* `05_tools/06_context_aware_tools.py`
* `06_handoffs/04_handoff_inputs_with_pydantic.py`
* `06_handoffs/06_recommended_handoff_callback.py`

---

*This project showcases advanced orchestration design, complex handoff flows, and real-world routing intelligence—representing my next milestone in mastering the OpenAI Agents SDK.*
