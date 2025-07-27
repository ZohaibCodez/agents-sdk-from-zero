# 10_multiple_agents — Multi-Agent Orchestration Patterns

This module demonstrates advanced multi-agent orchestration patterns using the OpenAI Agents SDK. It covers LLM-driven, code-driven, and hybrid orchestration strategies for real-world domains like fitness planning and university tasks, with robust error handling, structured outputs, and educational code.

---

## 🚀 Learning Goals

- Understand LLM-driven, code-driven, and hybrid orchestration strategies
- Build and chain specialized agents for complex workflows
- Use structured output models for planning, evaluation, and task routing
- Implement robust error handling and fallback logic in multi-agent systems
- Analyze and compare orchestration strategies for real-world domains

---

## 📂 Contents

| Script                                 | Description                                                                                   |
|----------------------------------------|-----------------------------------------------------------------------------------------------|
| `01_llm_orchestrated_agents.py`        | LLM-driven orchestration for fitness planning with multiple specialized agents and handoffs.   |
| `02_code_orchestrated_agents.py`       | Code-driven orchestration for university tasks: classification, planning, summary, MCQ, evaluation. |
| `03_hybrid_orchestration.py`           | Hybrid orchestration with dynamic strategy selection, fallback, and advanced agent chaining.   |

---

## ⚡ Quick Start

1. **Set up environment variables:**
   - `GEMINI_API_KEY` (required for 01 and 02)
   - `MISTRAL_API_KEY` (required for 03)
   - See root `.env.example` for details

2. **Run a demo script:**
   ```bash
   # LLM-driven multi-agent orchestration (fitness)
   python modules/10_multiple_agents.py/01_llm_orchestrated_agents.py

   # Code-driven orchestration (university tasks)
   python modules/10_multiple_agents.py/02_code_orchestrated_agents.py

   # Hybrid orchestration with dynamic strategy
   python modules/10_multiple_agents.py/03_hybrid_orchestration.py
   ```

---

## 🗂️ File Summaries

### 01_llm_orchestrated_agents.py
- LLM-driven orchestration for fitness planning
- Specialized agents: workout, diet, supplement, orchestrator
- Tool usage, agent handoffs, robust error handling
- Multiple demo scenarios (fat loss, beginner, advanced)

### 02_code_orchestrated_agents.py
- Code-driven orchestration for university tasks
- Structured output models for classification, planning, summary, MCQ, evaluation
- Four orchestration patterns: routing, chaining, parallel, iterative improvement
- Specialized agents for each task and robust error handling

### 03_hybrid_orchestration.py
- Hybrid orchestration for university tasks
- Dynamic strategy selection: LLM-driven, code-driven, hybrid
- Specialized agents for planning, notes, motivation, tools, execution
- Robust fallback, error handling, and educational output

---

## 🧩 Key Concepts

- **LLM-Orchestrated Agents:** Use LLMs to route, chain, and coordinate agent workflows
- **Code-Orchestrated Agents:** Use Python logic for explicit agent orchestration and control
- **Hybrid Orchestration:** Combine LLM and code strategies for adaptive, robust workflows
- **Structured Outputs:** Use Pydantic models for planning, evaluation, and task routing
- **Fallback & Error Handling:** Build resilient systems with robust error and fallback logic

---

## 🛠️ Environment & Configuration

- Requires valid API keys for Gemini (Google) and/or Mistral (OpenRouter)
- See `.env.example` in project root for all required/optional variables
- Scripts use `dotenv` for environment loading

---

## 📚 References

- [OpenAI Agents SDK Documentation](https://github.com/openai/openai-agents-sdk)
- [Pydantic Models](https://docs.pydantic.dev/)

---

## 🏆 Learning Outcomes

- Build advanced, real-world multi-agent orchestration systems
- Compare and analyze orchestration strategies for different domains
- Implement robust, production-ready agentic workflows
