# ⚡ 02_runner — Agent Execution & RunResult Exploration

_A hands-on exploration of the OpenAI Agents SDK Runner, RunResult object properties, and agent execution patterns._

---

## 📚 About This Module

This folder explores the **OpenAI Agents SDK Runner** - the core execution engine that brings agents to life. Here, I dive deep into agent execution patterns, RunResult object exploration, error handling, and understanding the complete lifecycle of agent runs. Each script demonstrates practical execution scenarios with detailed property analysis.

---

## 🗂️ Contents

| File/Notebook                  | Description                                                                 |
| ------------------------------ | --------------------------------------------------------------------------- |
| `01_run.py`                    | Comprehensive guide: basic agent execution, RunResult object exploration, and property analysis. |

---

## 🏁 Quick Start

> **Prerequisites:**  
> - Python environment with dependencies installed (`uv sync`)  
> - `.env` file with your API key(s) in the project root

### Run the Basic Agent Execution Example

```bash
uv run modules/02_runner/01_run.py
```

---

## 📄 File Summaries

### `01_run.py`
- **Core Focus**: Agent execution using the Runner and comprehensive RunResult exploration
- **Key Features**:
  - Basic agent creation and execution with the Runner
  - Complete RunResult object property analysis
  - Error handling for agent execution
  - Educational exploration of all RunResult properties
- **Learning Outcomes**:
  - Understand how to execute agents using the Runner
  - Master RunResult object structure and properties
  - Learn proper error handling patterns
  - Explore agent execution lifecycle

---

## 🔍 RunResult Object Deep Dive

The `01_run.py` script provides a comprehensive exploration of the RunResult object, including:

### Core Properties
- **`final_output`**: The agent's final response
- **`raw_responses`**: All model responses during execution
- **`input`**: The original user input
- **`new_items`**: Items added during execution

### Context & State
- **`context_wrapper`**: Execution context and state
- **`last_agent`**: The final agent in the execution chain
- **`last_response_id`**: Unique identifier for the last response

### Guardrails & Safety
- **`input_guardrail_results`**: Input validation results
- **`output_guardrail_results`**: Output safety check results

### Utility Methods
- **`to_input_list()`**: Convert result to input list format
- **`final_output_as(str)`**: Type-safe output conversion

---

## 🛠️ Configuration

- **API Keys:**  
  Place your OpenRouter API key in the root `.env` file:
  ```
  OPENROUTER_API_KEY=sk-...
  ```

- **Dependencies:**  
  Managed at the project root. Use `uv sync` to install.

---

## 🧭 Learning Goals

- Master agent execution using the Runner
- Understand RunResult object structure and properties
- Learn proper error handling for agent execution
- Explore agent execution lifecycle and state management
- Build confidence in debugging and analyzing agent runs

---

## 🔗 References & Further Reading

- [OpenAI Agents SDK Documentation](https://openai.github.io/openai-agents-python/quickstart/)
- [Project Root README](../../README.md) for setup, philosophy, and structure

---

## 🗃️ Related Projects & Showcases

- **[01_agents Module](../01_agents/)**: Foundation agent patterns, context, and structured outputs
- **[Projects Folder](../projects/)**: Real-world agentic applications and advanced workflows

---

**Learning by executing, one run at a time!**
