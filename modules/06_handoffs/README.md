# 06_handoffs — Advanced Agent Handoffs & Input Filtering

This module demonstrates advanced handoff patterns, input filtering, and callback mechanisms using the OpenAI Agents SDK. It covers multi-agent orchestration, privacy-preserving handoffs, structured data transfer, and recommended prompt engineering for robust, real-world agent systems.

---

## 🚀 Learning Goals
- Understand multi-agent handoff orchestration and routing
- Implement custom callbacks for handoff events
- Use structured data (Pydantic models) for handoff inputs
- Apply input filters for privacy, redaction, and summarization
- Engineer recommended prompts for better agent handoff understanding
- Explore robust error handling and professional output formatting

---

## 📂 Contents

| Script                                 | Description                                                                                   |
|----------------------------------------|-----------------------------------------------------------------------------------------------|
| `01_basic_handoffs.py`                 | Multi-agent handoff demo (triage to Enrollment, Finance, Tech). Streaming, error handling.     |
| `02_handoffs_with_callbacks.py`        | Handoffs with custom tool names, input types, and callback functions for each department.      |
| `03_handoff_inputs.py`                 | Structured handoff using Pydantic models, data validation, and async callbacks.                |
| `04_handoff_input_filter_basics.py`    | Demonstrates input filters to control what context is passed (with/without tool logs).         |
| `05_handoff_input_filters.py`          | Advanced input filters for privacy (FERPA/GDPR), redaction, summarization, and compliance.     |
| `06_recommended_prompts.py`            | Shows recommended prompt patterns for handoff clarity and agent understanding.                 |

---

## ⚡ Quick Start

1. **Set up environment variables:**
   - `GEMINI_API_KEY` (required for most scripts)
   - `MISTRAL_API_KEY` (required for 04_handoff_input_filter_basics.py)
   - See root `.env.example` for details

2. **Run a demo script:**
   ```bash
   # Example: Run the basic handoff demo
   python modules/06_handoffs/01_basic_handoffs.py
   ```

3. **Explore other scripts:**
   ```bash
   python modules/06_handoffs/02_handoffs_with_callbacks.py
   python modules/06_handoffs/03_handoff_inputs.py
   python modules/06_handoffs/04_handoff_input_filter_basics.py
   python modules/06_handoffs/05_handoff_input_filters.py
   python modules/06_handoffs/06_recommended_prompts.py
   ```

---

## 🗂️ File Summaries

### 01_basic_handoffs.py
- University support triage bot hands off to Enrollment, Finance, or Tech agents
- Demonstrates agent chaining, streaming, and robust output formatting
- Shows how to define handoff agents and route based on user input

### 02_handoffs_with_callbacks.py
- Adds custom tool names and callback functions for each handoff
- Demonstrates how to trigger side effects or logging on handoff events
- Shows use of Pydantic models for structured IT issue handoff

### 03_handoff_inputs.py
- Advanced structured handoff with multiple specialized agents
- Uses Pydantic models for data validation and input structure
- Async callbacks for each handoff, with visually distinct output
- Robust error handling and test/demo cases

### 04_handoff_input_filter_basics.py
- Demonstrates input filters to control what context is passed to the next agent
- Shows difference between standard handoff (with tool logs) and filtered handoff (tool calls removed)
- Educational comments and robust error handling

### 05_handoff_input_filters.py
- Advanced input filters for privacy, redaction, and summarization
- Custom filters for FERPA/GDPR compliance and privacy officer handoff
- Multiple handoff types: raw, redacted, summarized, privacy
- Professional output and robust test cases

### 06_recommended_prompts.py
- Shows recommended prompt patterns for better handoff understanding
- Demonstrates use of `RECOMMENDED_PROMPT_PREFIX` and helper functions
- Example of triage agent with clear handoff instructions and specialist teams

---

## 🧩 Key Concepts

- **Agent Handoffs:** Seamless transfer of context and control between specialized agents.
- **Callbacks:** Custom functions triggered on handoff events for logging, side effects, or data processing.
- **Input Filters:** Functions to redact, summarize, or transform context before handoff (privacy, compliance, clarity).
- **Structured Inputs:** Use of Pydantic models for validating and structuring handoff data.
- **Prompt Engineering:** Recommended prompt patterns for clarity, transfer rationale, and agent expectations.

---

## 🛠️ Environment & Configuration
- Requires valid API keys for Gemini (Google) and/or Mistral (OpenRouter)
- See `.env.example` in project root for all required/optional variables
- Scripts use `dotenv` for environment loading

---

## 📚 References
- [OpenAI Agents SDK Documentation](https://github.com/openai/openai-agents-sdk)
- [Pydantic Models](https://docs.pydantic.dev/)
- [FERPA](https://www2.ed.gov/policy/gen/guid/fpco/ferpa/index.html), [GDPR](https://gdpr-info.eu/)

---

## 🏆 Learning Outcomes
- Build robust, privacy-aware multi-agent systems
- Implement real-world handoff and escalation flows
- Engineer clear, maintainable, and professional agent orchestration code
