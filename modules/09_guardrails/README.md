# 09_guardrails — Input & Output Guardrails, Safety, and Compliance

This module demonstrates how to implement input and output guardrails, safety checks, and compliance patterns using the OpenAI Agents SDK. It covers robust guardrail tripwire handling, agent-based guardrails, and exception management for building safe, production-ready agentic systems.

---

## 🚀 Learning Goals
- Understand input and output guardrail patterns
- Implement guardrail tripwires and exception handling
- Build agent-based guardrails for complex workflows
- Apply safety, privacy, and compliance checks to agent interactions
- Debug and report guardrail failures in a user-friendly way

---

## 📂 Contents

| Script                              | Description                                                                 |
|-------------------------------------|-----------------------------------------------------------------------------|
| `01_basic_input_guardrail.py`       | Demonstrates input guardrails (e.g., only allow movie-related questions).   |
| `02_basic_output_guardrail.py`      | Demonstrates output guardrails (e.g., block sensitive words in responses).  |
| `03_guardrail_exceptions.py`        | Shows exception handling for guardrail tripwires and robust error reporting.|
| `04_agent_based_guardrails.py`      | Advanced agent-based guardrails for multi-step and contextual safety.       |

---

## ⚡ Quick Start

1. **Set up environment variables:**
   - `GEMINI_API_KEY` (required for all scripts)
   - See root `.env.example` for details

2. **Run a demo script:**
   ```bash
   # Example: Run the basic input guardrail demo
   python modules/09_guardrails/01_basic_input_guardrail.py
   ```

3. **Explore other scripts:**
   ```bash
   python modules/09_guardrails/02_basic_output_guardrail.py
   python modules/09_guardrails/03_guardrail_exceptions.py
   python modules/09_guardrails/04_agent_based_guardrails.py
   ```

---

## 🗂️ File Summaries

### 01_basic_input_guardrail.py
- Demonstrates input guardrails that only allow movie-related questions
- Shows how to catch and report input guardrail tripwires
- Educational, robust, and production-ready example

### 02_basic_output_guardrail.py
- Demonstrates output guardrails that block sensitive words in agent responses
- Shows how to catch and report output guardrail tripwires
- Robust error handling and user-friendly output

### 03_guardrail_exceptions.py
- Shows exception handling for guardrail tripwires
- Demonstrates robust error reporting and debugging patterns
- Educational comments and clear output formatting

### 04_agent_based_guardrails.py
- Advanced agent-based guardrails for multi-step and contextual safety
- Demonstrates how to chain guardrails and apply them in complex workflows
- Robust, production-ready patterns for compliance and privacy

---

## 🧩 Key Concepts

- **Input Guardrails:** Validate and restrict user input before agent processing
- **Output Guardrails:** Check and filter agent responses for safety and compliance
- **Guardrail Tripwires:** Mechanism for blocking unsafe or non-compliant interactions
- **Exception Handling:** Robust error reporting for guardrail failures
- **Agent-Based Guardrails:** Apply guardrails contextually across multi-step workflows

---

## 🛠️ Environment & Configuration
- Requires valid API key for Gemini (Google)
- See `.env.example` in project root for all required/optional variables
- Scripts use `dotenv` for environment loading

---

## 📚 References
- [OpenAI Agents SDK Documentation](https://github.com/openai/openai-agents-sdk)
- [Guardrails in OpenAI Agents SDK](https://platform.openai.com/docs/agents/guardrails)

---

## 🏆 Learning Outcomes
- Build safe, privacy-aware agentic systems
- Implement robust guardrail and compliance patterns
- Debug and report guardrail failures in production
- Create professional, user-friendly safety mechanisms
