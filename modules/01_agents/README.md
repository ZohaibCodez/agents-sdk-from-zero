# 🚀 01_agents — Agent Fundamentals & Context Handling

_A hands-on, example-driven exploration of the OpenAI Agents SDK's core agent patterns and context management._

---

## 📚 About This Module

This folder is part of my journey to master the **OpenAI Agents SDK**. Here, I experiment with foundational agent concepts and context passing. Each script is a step in my learning-by-doing approach, with code, explanations, and real outputs.

---

## 🗂️ Contents

| File                  | Description                                                      |
| --------------------- | ----------------------------------------------------------------|
| `03_agents_ins.py`    | Script: agent instantiation and instruction examples.             |
| `04_agent_context.py` | Script: demonstrates context-aware agents with callable instructions. |

---

## 🏁 Quick Start

> **Prerequisites:**  
> - Python environment with dependencies installed (`uv sync`)  
> - `.env` file with your API key(s) in the project root

### Run Agent Instantiation Example

```bash
uv run modules/01_agents/03_agents_ins.py
```

### Run Context-Aware Agent Example

```bash
uv run modules/01_agents/04_agent_context.py
```

---

## 📄 File Summaries

### `03_agents_ins.py`
- Agent instantiation and instruction examples
- Demonstrates different ways to configure and run agents

### `04_agent_context.py`
- Passing user context to agents
- Using a callable to generate instructions based on context
- Running the agent and printing both context and output

---

## 🛠️ Configuration

- **API Keys:**  
  Place your OpenAI API key in the root `.env` file:
  ```
  OPENAI_API_KEY=sk-...
  ```

- **Dependencies:**  
  Managed at the project root. Use `uv sync` to install.

---

## 🧭 Learning Goals

- Understand agent creation and configuration
- Explore context-aware instructions
- Build a foundation for advanced agentic workflows

---

## 🔗 References & Further Reading

- [OpenAI Agents SDK Documentation](https://openai.github.io/openai-agents-python/quickstart/)
- [Project Root README](../../README.md) for setup, philosophy, and structure

---

**Learning by building, one agent at a time!**
