# 📊 03_results — RunResult Basics & Results Handling

_A focused module for exploring the OpenAI Agents SDK RunResult object, its attributes, and practical results handling patterns._

---

## 📚 About This Module

This folder demonstrates foundational usage of the **RunResult** object from the OpenAI Agents SDK. It covers how to access, inspect, and utilize RunResult attributes and methods for building robust agentic workflows and multi-turn conversations.

---

## 🗂️ Contents

| File/Notebook                  | Description                                                                 |
| ------------------------------ | --------------------------------------------------------------------------- |
| `01_run_result_basics.py`      | Demonstrates basic RunResult attributes, methods, and multi-turn input handling. |

---

## 🏁 Quick Start

> **Prerequisites:**  
> - Python environment with dependencies installed (`uv sync`)  
> - `.env` file with your API key(s) in the project root

### Run the RunResult Basics Example

```bash
uv run modules/03_results/01_run_result_basics.py
```

---

## 📄 File Summaries

### `01_run_result_basics.py`
- **Core Focus**: Exploring RunResult attributes and methods
- **Key Features**:
  - Accessing `final_output`, `last_agent`, `new_items`, and `input`
  - Using `to_input_list()` for multi-turn conversations
  - Inspecting and printing RunResult content and types
  - Demonstrates how to build on RunResult for follow-up turns
- **Learning Outcomes**:
  - Understand the structure and purpose of RunResult
  - Learn to access and use RunResult attributes
  - Master multi-turn input handling with `to_input_list()`
  - Build confidence in debugging and analyzing agent results

---

## 🧭 Learning Goals

- Master the basics of RunResult usage
- Learn to inspect and utilize RunResult attributes
- Build multi-turn agentic workflows using RunResult
- Develop robust debugging and results analysis skills

---

## 🔗 References & Further Reading

- [OpenAI Agents SDK Documentation](https://openai.github.io/openai-agents-python/results/)
- [Project Root README](../../README.md) for setup, philosophy, and structure

---

**Learning by inspecting, one result at a time!** 