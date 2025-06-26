# 🤖 My OpenAI Agents SDK Learning Journey

_A hands-on, teaching-by-doing path to mastering the OpenAI Agents SDK._

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
│   │   ├── 🐍 01_hello_agent.py
│   │   ├── 📓 03_agents_instructions.ipynb
│   │   ├── 🐍 04_agent_context.py
│   │   ├── 📓 05_immutable_context.ipynb
│   │   ├── 🐍 06_structure.py
│   │   ├── 🐍 07_non_strict_output_type.py
│   │   └── 📄 README.md
│   └── 📁 02_runner/   # (scaffolded, currently empty)
├── 🚫 .gitignore
├── 📄 .python-version
├── 🐍 main.py
├── 📋 pyproject.toml
├── 📖 README.md
└── 🔒 uv.lock
└── 📄 .env.example
```

> **Note:** Each module (e.g., `01_agents/`) contains its own README for detailed explanations, code summaries, and learning notes.

---

## 📈 Progress Tracker

| Module                 | Status                                                      |
| ---------------------- | ----------------------------------------------------------- |
| 🚀 01_agents           | 🚧 In Progress: Core agent patterns, context, instructions, structured outputs |
| ⚡ 02_runner           | 🚧 In Progress (scaffolded, content coming soon)             |
| 📊 03-handling-results | ⏳ Coming Up                                                |
| 🔧 04-tool-integration | ⏳ Coming Up                                                |
| ...                    | ...                                                         |

---

## 🆕 Recent Advanced Additions

- **Structured Outputs & Schemas:** See `modules/01_agents/06_structure.py` for strict/non-strict schema patterns, Pydantic models, and validation.
- **Advanced Context Management:** See `modules/01_agents/05_immutable_context.ipynb` for stateful agent context and function tools.
- **Instruction Patterns:** See `modules/01_agents/03_agents_instructions.ipynb` for static, dynamic, and callable agent instructions.

---

## 🛠️ Setup & Installation

### 📦 Prerequisites

- Python (version specified in `.python-version`)
- [uv](https://github.com/astral-sh/uv) package manager installed
- `.env` file with OpenAI/Gemini API keys

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
Master the OpenAI Agents SDK from foundation to production-ready apps.
**📚 Philosophy:**
Learn deeply. Build consistently. Teach simply.

---

## 🔗 Connect & Learn Together

- 📢 Follow this repo for updates
- 🤝 Let's build the Agentic future, one concept at a time

---

**📚 Learning Never Stops** – This repository is a living record of my OpenAI Agents SDK mastery. Join me on the journey!

_Last Updated: 2025-06-27_
