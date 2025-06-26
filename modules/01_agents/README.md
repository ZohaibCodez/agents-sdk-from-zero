# 🚀 01_agents — Agent Fundamentals & Context Handling

_A hands-on, example-driven exploration of the OpenAI Agents SDK's core agent patterns, context management, and structured outputs._

---

## 📚 About This Module

This folder is part of my journey to master the **OpenAI Agents SDK**. Here, I experiment with foundational agent concepts, context passing, advanced instruction patterns, and structured outputs. Each script and notebook is a step in my learning-by-doing approach, with code, explanations, and real outputs.

---

## 🗂️ Contents

| File/Notebook                  | Description                                                                 |
| ------------------------------ | --------------------------------------------------------------------------- |
| `01_hello_agent.py`            | Minimal example: create and run a basic agent with static instructions.      |
| `03_agents_instructions.ipynb` | Interactive notebook: explores agent instructions (static, dynamic, async).  |
| `04_agent_context.py`          | Script: demonstrates context-aware agents with callable instructions.        |
| `05_immutable_context.ipynb`   | Notebook: advanced context/state management and function tools.              |
| `06_structure.py`              | Comprehensive guide: strict/non-strict schemas, structured outputs, Pydantic models, validation, and advanced agent output patterns. |
| `07_non_strict_output_type.py` | Placeholder for future non-strict output type experiments.                   |

---

## 🏁 Quick Start

> **Prerequisites:**  
> - Python environment with dependencies installed (`uv sync`)  
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

### Structured Output & Advanced Examples

```bash
uv run modules/01_agents/06_structure.py
```

---

## 📄 File Summaries

### `01_hello_agent.py`
- Minimal agent example using Gemini
- Loads API keys from `.env`
- Instantiates a basic agent with static instructions
- Runs the agent and prints the output

### `03_agents_instructions.ipynb`
- Explores agent instructions: static, dynamic, callable, and context-aware
- Practical code cells and output for each pattern

### `04_agent_context.py`
- Passing user context to agents
- Using a callable to generate instructions based on context
- Running the agent and printing both context and output

### `05_immutable_context.ipynb`
- Advanced context/state management
- Function tools for manipulating context (save, get, clear preferences)
- Integration with the agent loop and tool usage
- Example: managing user preferences with agent tools

### `06_structure.py`
- Comprehensive guide to structured outputs
- Strict vs non-strict schemas, Pydantic models, validation
- Advanced agent output patterns and use cases

### `07_non_strict_output_type.py`
- Placeholder for future non-strict output type experiments

---

## 🛠️ Configuration

- **API Keys:**  
  Place your OpenAI or Gemini API key in the root `.env` file:
  ```
  OPENAI_API_KEY=sk-...
  GEMINI_API_KEY=...
  ```

- **Dependencies:**  
  Managed at the project root. Use `uv sync` to install.

---

## 🧭 Learning Goals

- Understand agent creation and configuration
- Explore dynamic and context-aware instructions
- Learn to manage agent state and context
- Master structured outputs and schema validation
- Build a foundation for advanced agentic workflows

---

## 🔗 References & Further Reading

- [OpenAI Agents SDK Documentation](https://openai.github.io/openai-agents-python/quickstart/)
- [Project Root README](../../README.md) for setup, philosophy, and structure

---

**Learning by building, one agent at a time!**
