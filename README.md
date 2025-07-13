# 🤖 My OpenAI Agents SDK Learning Journey

_A hands-on, teaching-by-doing path to mastering the OpenAI Agents SDK._

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![OpenAI Agents SDK](https://img.shields.io/badge/OpenAI%20Agents%20SDK-Latest-green.svg)
![Learning Status](https://img.shields.io/badge/status-learning%20in%20progress-orange.svg)
![Last Commit](https://img.shields.io/github/last-commit/ZohaibCodez/agents-sdk-from-zero)
![GitHub Stars](https://img.shields.io/github/stars/ZohaibCodez/agents-sdk-from-zero)
![GitHub Forks](https://img.shields.io/github/forks/ZohaibCodez/agents-sdk-from-zero)

---

## 📖 About This Repository

This repository documents my comprehensive learning journey through the **OpenAI Agents SDK**, following a hands-on, teaching-by-doing approach. As I master each concept, I create my own explanations, examples, and projects to solidify understanding and help fellow learners.

---

## 🎯 Learning Philosophy

> "The best way to learn is to teach." — _Richard Feynman_

I believe in learning through active creation and explanation. This repo captures not just what I've learned, but **how** I learned it — including challenges, breakthroughs, and real applications.

---

## 🙏 Acknowledgments & Attribution

This journey is inspired by [**@mjunaidca's agents-sdk-decoded**](https://github.com/mjunaidca/agents-sdk-decoded) — a brilliant expert-level educational resource.

### 🔍 What Makes My Approach Different

- 📝 Personal explanations in my own words
- 🎯 Focused on **practical understanding**
- 🚀 Live examples built step-by-step
- 📊 Documenting the **learning process**
- 💡 Beginner-friendly + real-world mindset

---

## 🧭 Learning Objectives

By the end of this journey, I aim to:

- ✅ Master the OpenAI Agents SDK from fundamentals to advanced patterns
- ✅ Build production-ready agentic applications
- ✅ Understand the **DACA** framework for scalable systems
- ✅ Create developer-friendly documentation
- ✅ Help others learn agentic AI — one concept at a time

---

## 🗂️ Repository Structure

```
.
├── 📁 modules/
│   ├── 📁 01_agents/
│   ├── 📁 02_runner/
│   ├── 📁 03_results/
│   ├── 📁 04_stream/
│   ├── 📁 05_tools/
│   ├── 📁 06_handoffs/           # NEW: Advanced handoff, input filtering, callbacks, prompt engineering
│   ├── 📁 07_lifecycle/          # NEW: Agent lifecycle hooks, monitoring, analytics
│   ├── 📁 08_exceptions/         # In Progress: Exception handling, error flows
│   ├── 📁 09_guardrails/         # In Progress: Guardrails, input validation, safety
│   └── 📁 projects/
│       ├── 📁 01_fantasy_world_generator/
│       ├── 📁 02_university_helpdesk_orchestration/   # NEW: Advanced multi-agent orchestration project
│       └── 📄 README.md
├── 🚫 .gitignore
├── 📄 .python-version
├── 🐍 main.py
├── 📋 pyproject.toml
├── 📖 README.md
├── 🔒 uv.lock
├── 📄 .env.example
├── 📄 LICENSE
```

> **Note:** Each module and project contains its own README for detailed explanations, code summaries, and learning notes.

---

## 📈 Progress Tracker

| Module/Section          | Status                                                      |
| ----------------------- | ----------------------------------------------------------- |
| 🚀 01_agents            | ✅ **Completed** Core agent patterns, context, instructions, structured outputs, advanced orchestration, tool behavior control, agent handoffs |
| ⚡ 02_runner            | ✅ **Completed** Agent execution, RunResult exploration, execution lifecycle, synchronous execution, streaming patterns, real-time event processing, advanced context, handoffs, chat, exceptions |
| 📊 03_results           | ✅ **Completed** RunResult basics, results handling, and multi-turn conversation management |
| 📡 04_stream            | ✅ **Completed** Advanced streaming patterns, real-time events, token-by-token streaming, run item events, and streaming with agent handoffs |
| 🔧 05_tools             | ✅ **Completed** Function tools, custom configurations, agents as tools, error handling, context-aware tools, advanced tool choice patterns |
| 🤝 06_handoffs          | ✅ **Completed** Advanced handoff orchestration, input filtering, callbacks, structured handoff data, prompt engineering |
| 🔄 07_lifecycle         | ✅ **Completed** Agent lifecycle hooks, monitoring, analytics, session management, performance tracking |
| 🚨 08_exceptions        | 🚧 **In Progress** Exception handling, error flows, advanced error management |
| 🛡️ 09_guardrails        | 🚧 **In Progress** Guardrails, input validation, safety, compliance |
| 🧩 Projects             | 🚀 Started: Real-world agentic applications and showcases    |
| ...                    | ...                                                         |

---

## 🗃️ Projects

The `modules/projects/` folder contains real-world, showcase projects that demonstrate advanced agentic workflows, tool integrations, and creative applications using the OpenAI Agents SDK.

### Featured Projects

- **[Fantasy World Generator](modules/projects/01_fantasy_world_generator/)**: An interactive multi-agent system that builds a fantasy world, including magic systems, maps, lore, and magical creatures. Demonstrates agent orchestration, agents-as-tools, streaming, and creative AI workflows.
- **[University Smart Helpdesk Orchestrator](modules/projects/02_university_helpdesk_orchestration/)**: A real-world multi-agent system simulating a university helpdesk. Features advanced orchestration, context-aware routing, escalation handling, streaming UX, and smart agent handoffs. Demonstrates context models, escalation logic, and real-time agent transitions.

See each project's README for setup and usage instructions.

---

## 🆕 Recent Advanced Additions

- **Agent Handoffs & Input Filtering:** See `modules/06_handoffs/` for advanced handoff orchestration, input filtering, callbacks, and prompt engineering.
- **Agent Lifecycle Hooks & Monitoring:** See `modules/07_lifecycle/` for RunHooks, AgentHooks, session management, and analytics.
- **Guardrails & Exception Handling:** See `modules/08_exceptions/` and `modules/09_guardrails/` for in-progress work on error management and safety.
- **University Helpdesk Orchestration Project:** See `modules/projects/02_university_helpdesk_orchestration/` for advanced multi-agent orchestration, escalation, and context-aware routing.
- **All previous advanced streaming, tool integration, and orchestration features remain available in their respective modules.**

---

## 🛠️ Setup & Installation

### 📦 Prerequisites

- Python (version specified in `.python-version`)
- [uv](https://github.com/astral-sh/uv) package manager installed
- `.env` file with OpenAI/Gemini/OpenRouter API keys

### 🚀 Quick Setup

```bash
# Clone the repository
git clone https://github.com/ZohaibCodez/agents-sdk-from-zero.git
cd agents-sdk-from-zero

# Initialize environment
uv venv
.venv\Scripts\activate   # For Windows
# or
source .venv/bin/activate   # For macOS/Linux

# Install dependencies
uv sync

# Copy environment variables
cp .env.example .env # Then add your API key inside
```

---

## 🔧 Dev Environment Overview

- **Python version**: Managed by `.python-version`
- **Dependency manager**: `uv` (`uv add`, `uv sync`)
- **Virtual environment**: `.venv/`
- **API secrets**: Stored in `.env` file

---

## 📈 Learning Methodology

1. **Understand** – Study SDK internals and examples
2. **Apply** – Build simplified working demos
3. **Explain** – Write markdown explanations and insights
4. **Extend** – Build real tools and integrations
5. **Repeat** – Keep iterating and improving

---

## 🤝 Contributing & Feedback

This is a **personal learning repo**, but:

- 💬 **Ask questions** via GitHub Issues
- ✍️ **Suggest improvements** if you spot unclear parts
- 🌟 **Star the repo** if it helps — it motivates me to keep going!

---

**🎯 Final Goal:**
Master the OpenAI Agents SDK from foundation to production-ready apps.<br>
**📚 Philosophy:**
Learn deeply. Build consistently. Teach simply.

---

## 🔗 Connect & Learn Together

- 📢 Follow this repo for updates
- 🤝 Let's build the Agentic future, one concept at a time

---

**📚 Learning Never Stops** – This repository is a living record of my OpenAI Agents SDK mastery. Join me on the journey!

_Last Updated: 2025-07-13_
