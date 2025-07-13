# 07_lifecycle — Agent Lifecycle Hooks & Monitoring

This module demonstrates agent lifecycle monitoring and hooks using the OpenAI Agents SDK. It covers RunHooks for tracking agent execution, AgentHooks for individual agent monitoring, and advanced analytics for performance optimization and debugging.

---

## 🚀 Learning Goals
- Understand RunHooks for tracking agent execution lifecycle
- Implement AgentHooks for individual agent monitoring and analytics
- Monitor tool usage, handoffs, and performance metrics
- Build advanced analytics and reporting systems
- Debug and optimize agent performance through lifecycle tracking
- Implement session management and response scoring

---

## 📂 Contents

| Script                                 | Description                                                                                   |
|----------------------------------------|-----------------------------------------------------------------------------------------------|
| `01_run_lifecycle_hooks.py`            | Simple email workflow demonstrating RunHooks basics. Tracks agent starts, ends, handoffs, and tool usage. |
| `02_agent_lifecycle_hooks.py`          | Advanced AgentHooks with three levels: Beginner, Intermediate, and Advanced monitoring with session tracking, performance analytics, and response scoring. |

---

## ⚡ Quick Start

1. **Set up environment variables:**
   - `GEMINI_API_KEY` (required for both scripts)
   - See root `.env.example` for details

2. **Run the basic lifecycle demo:**
   ```bash
   # Run the simple email workflow with RunHooks
   python modules/07_lifecycle/01_run_lifecycle_hooks.py
   ```

3. **Run the advanced lifecycle demo:**
   ```bash
   # Run the advanced AgentHooks with multiple monitoring levels
   python modules/07_lifecycle/02_agent_lifecycle_hooks.py
   ```

---

## 🗂️ File Summaries

### 01_run_lifecycle_hooks.py
- Simple email workflow demonstrating RunHooks fundamentals
- Tracks agent starts, ends, handoffs, and tool usage
- Implements `SimpleEmailHooks` class with event logging
- Shows basic metrics collection and summary reporting
- Educational focus on understanding core lifecycle events

### 02_agent_lifecycle_hooks.py
- Advanced AgentHooks with three monitoring levels:
  - **Beginner**: Basic activation tracking and tool usage counting
  - **Intermediate**: Session management with detailed timing and tool analytics
  - **Advanced**: Comprehensive monitoring with response scoring and optimization notes
- Demonstrates session tracking, performance analytics, and response quality assessment
- Shows how to implement custom analytics and reporting systems

---

## 🧩 Key Concepts

- **RunHooks**: System-level hooks that track agent execution across the entire workflow
- **AgentHooks**: Individual agent-level hooks for monitoring specific agent behavior
- **Lifecycle Events**: Start, end, tool usage, and handoff events that can be monitored
- **Session Management**: Tracking individual agent sessions with timing and context
- **Performance Analytics**: Response time, tool usage patterns, and optimization insights
- **Response Scoring**: Quality assessment of agent outputs based on various metrics

---

## 🔧 Lifecycle Events

### RunHooks Events
- `on_agent_start`: Called when an agent begins processing
- `on_agent_end`: Called when an agent completes its task
- `on_handoff`: Called when one agent transfers to another
- `on_tool_start`: Called when an agent begins using a tool
- `on_tool_end`: Called when an agent finishes using a tool

### AgentHooks Events
- `on_start`: Agent-specific start event with context
- `on_end`: Agent-specific end event with output
- `on_tool_start`: Agent-specific tool usage start
- `on_tool_end`: Agent-specific tool usage completion
- `on_handoff`: Agent-specific handoff event

---

## 📊 Monitoring Levels

### Beginner Level
- Basic activation counting
- Simple tool usage tracking
- Processing time measurement
- Minimal session information

### Intermediate Level
- Detailed session management
- Tool usage timing and results
- Input/output previews
- Comprehensive session reporting

### Advanced Level
- Response quality scoring
- Performance optimization notes
- Multi-session analytics
- Advanced debugging capabilities

---

## 🛠️ Environment & Configuration
- Requires valid API key for Gemini (Google)
- See `.env.example` in project root for all required/optional variables
- Scripts use `dotenv` for environment loading

---

## 📚 References
- [OpenAI Agents SDK Documentation](https://github.com/openai/openai-agents-sdk)
- [Agent Lifecycle Management](https://github.com/openai/openai-agents-sdk/blob/main/docs/concepts/lifecycle.md)

---

## 🏆 Learning Outcomes
- Build comprehensive agent monitoring and analytics systems
- Implement performance tracking and optimization strategies
- Debug complex multi-agent workflows effectively
- Create professional-grade agent lifecycle management tools
