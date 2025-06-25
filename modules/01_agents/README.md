# 🚀 01_agents — Agent Fundamentals & Context Handling

_A hands-on, example-driven exploration of the OpenAI Agents SDK's core agent patterns, context management, and dynamic instruction techniques._

---

## 📚 About This Module

This folder is part of my journey to master the **OpenAI Agents SDK**. Here, I experiment with foundational agent concepts, context passing, and advanced instruction patterns. Each script and notebook is a step in my learning-by-doing approach, with code, explanations, and real outputs.

---

## 🗂️ Contents

| File/Notebook                  | Description                                                                 |
| ------------------------------ | --------------------------------------------------------------------------- |
| `01_hello_agent.py`            | Minimal example: create and run a basic agent with static instructions.      |
| `03_agents_instructions.ipynb` | Interactive notebook: explores agent instructions (static, dynamic, async).  |
| `04_agent_context.py`          | Script: demonstrates context-aware agents with callable instructions.        |
| `05_immutable_context.ipynb`   | Notebook: advanced context/state management and function tools.              |

---

## 🏁 Quick Start

> **Prerequisites:**  
> - Python environment with dependencies installed (`uv sync` or `pip install -r requirements.txt`)  
> - `.env` file with your API key(s) in the project root

### Run a Basic Agent Example

```bash
uv run modules/01_agents/01_hello_agent.py
```

### Run a Context-Aware Agent Example

```bash
uv run modules/01_agents/04_agent_context.py
```

### Explore Interactive Notebooks

Open the notebooks in your favorite Jupyter environment:

- `03_agents_instructions.ipynb`
- `05_immutable_context.ipynb`

---

## 📄 File Summaries

### `01_hello_agent.py`
- Loads API keys from `.env`
- Disables tracing for privacy
- Instantiates a basic agent with static instructions
- Runs the agent on a simple prompt and prints the output

### `03_agents_instructions.ipynb`
- Static and dynamic agent instructions
- Callable and async instruction functions
- Context-aware instruction generation
- Stateful instruction patterns
- Practical code cells and output for each pattern

### `04_agent_context.py`
- Passing user context to agents
- Using a callable to generate instructions based on context
- Running the agent and printing both context and output

### `05_immutable_context.ipynb`
- Custom context objects and state management
- Function tools for manipulating context (save, get, clear preferences)
- Integration with the agent loop and tool usage
- Example: managing user preferences with agent tools

---

## 🛠️ Configuration

- **API Keys:**  
  Place your OpenAI or Gemini API key in the root `.env` file:
  ```
  OPENAI_API_KEY=sk-...
  # or for Gemini (if supported by your setup)
  GEMINI_API_KEY=...
  ```

- **Dependencies:**  
  Managed at the project root. Use `uv sync` or `pip install -r requirements.txt`.

---

## 🧭 Learning Goals

- Understand agent creation and configuration
- Explore dynamic and context-aware instructions
- Learn to manage agent state and context
- Build a foundation for advanced agentic workflows

---

## 🔗 References & Further Reading

- [OpenAI Agents SDK Documentation](https://openai.github.io/openai-agents-python/quickstart/)
- [Project Root README](../../README.md) for setup, philosophy, and structure

---

**Learning by building, one agent at a time!**
