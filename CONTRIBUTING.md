# 🤝 Contributing to agents-sdk-from-zero

Welcome to my learning repo on the OpenAI Agents SDK!  
This project is part of my journey to master agentic AI by re-building and teaching each concept in a clean, professional way.

If you're also learning or want to contribute — whether fixing a typo or improving a demo — you're welcome here. Let's build this resource together! 🚀

---

## ✨ How You Can Contribute

### 🧠 If You're Learning Like Me:
- Ask questions by opening an issue
- Suggest improvements in explanations or examples
- Share your confusion — it's helpful to others too!
- Try running a demo and let me know if something's broken

### 🧰 If You're More Experienced:
- Refactor logic for clarity
- Suggest better design patterns
- Add comments or better prompt engineering ideas
- Add helpful test cases or tools

---

## 🔧 Getting Started

### Step 1: Fork and Clone the Repo
```bash
git clone https://github.com/YOUR-USERNAME/agents-sdk-from-zero.git
cd agents-sdk-from-zero
````

### Step 2: Setup Environment

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
cp .env.example .env       # Add your OpenAI API key in this file
```

---

## 📁 Folder Structure Guidelines

Please follow the style I’m using in each module:

```
📁 modules/
│   ├── 📁 01_agents/
│   │   ├── 🐍 01_hello_agent.py
│   │   ├── 🐍 02_hello_handoff.py
|   |   ├── README.md   # What you’ll learn here
│   ├── README.md               
```

Each folder = one concept = one module.

---

## 🧾 Code Style

* Use meaningful names for agents, contexts, and demos
* Add comments where logic might confuse beginners
* Keep examples focused: one demo, one concept
* Follow [PEP8](https://peps.python.org/pep-0008/) basics

---

## 📝 Commit Message Tips

Please write clear commit messages like:

```
Add guardrail test cases for sensitive input detection

- Covers normal query, risky input, and permission denial
```

---

## 🔁 Pull Request Checklist

Before opening a PR:

* ✅ Run the code (test if possible)
* ✅ Comment complex logic
* ✅ Mention what demo or feature it improves
* ✅ Link to any related issue or discussion

---

## 🐛 Found a Bug?

Open an issue and include:

* Input or query that caused it
* Error message or behavior
* What you expected
* Screenshot or traceback if possible

---

## 🧠 Why This Project Matters

This repo is not just code — it’s a roadmap for mastering agentic AI.
It follows the principle: **“Learn like a teacher. Build like a pro.”**
So every contribution, big or small, helps improve understanding.

---

## 📜 License

By contributing, you agree your code is licensed under the [MIT License](LICENSE).

---

Thank you for helping make this a better place to learn OpenAI Agents! 🙌
Whether you ask questions or fix code — it’s all a contribution.

*– Mr. Zohaib*

