# 📡 04_stream — Advanced Streaming Patterns & Real-Time Events

_A comprehensive exploration of streaming capabilities in the OpenAI Agents SDK, covering basic streaming, raw response events, run item events, and streaming with agent handoffs._

---

## 📚 About This Module

This folder demonstrates advanced streaming patterns and real-time event processing with the OpenAI Agents SDK. Here, I explore different levels of streaming - from basic text streaming to complex agent handoffs with real-time event monitoring. Each script showcases practical streaming scenarios with detailed event analysis and user-friendly output formatting.

---

## 🗂️ Contents

| File/Notebook                  | Description                                                                 |
| ------------------------------ | --------------------------------------------------------------------------- |
| `01_basic_streaming.py`        | Simple introduction to streaming responses with real-time text output. |
| `02_raw_response_events.py`    | Token-by-token streaming with raw response events and text delta analysis. |
| `03_run_item_events.py`        | Higher-level streaming with run item events, tool calls, and agent updates. |
| `04_streaming_with_handoffs.py`| Advanced streaming with agent handoffs, transitions, and real-time monitoring. |

---

## 🏁 Quick Start

> **Prerequisites:**  
> - Python environment with dependencies installed (`uv sync`)  
> - `.env` file with your API key(s) in the project root

### Run Basic Streaming Examples

```bash
# Basic streaming with real-time text output
uv run modules/04_stream/01_basic_streaming.py

# Token-by-token streaming with raw response events
uv run modules/04_stream/02_raw_response_events.py

# Higher-level streaming with run item events
uv run modules/04_stream/03_run_item_events.py

# Advanced streaming with agent handoffs
uv run modules/04_stream/04_streaming_with_handoffs.py
```

---

## 📄 File Summaries

### `01_basic_streaming.py`
- **Core Focus**: Simple introduction to streaming responses
- **Key Features**:
  - Basic streaming with `Runner.run_streamed()`
  - Real-time text output as it's generated
  - Event type monitoring and analysis
  - Comparison between streamed and final output
- **Learning Outcomes**:
  - Understand basic streaming patterns
  - Learn to monitor streaming events
  - Master real-time text output handling
  - Build confidence in streaming fundamentals

### `02_raw_response_events.py`
- **Core Focus**: Token-by-token streaming with raw response events
- **Key Features**:
  - Token-by-token text streaming using `ResponseTextDeltaEvent`
  - Raw response event analysis and statistics
  - Text delta collection and comparison
  - Event type counting and analysis
- **Learning Outcomes**:
  - Master token-by-token streaming
  - Understand raw response event structure
  - Learn to analyze streaming statistics
  - Build real-time text processing capabilities

### `03_run_item_events.py`
- **Core Focus**: Higher-level streaming with run item events and tool integration
- **Key Features**:
  - Higher-level event streaming (tool calls, messages, agent updates)
  - Tool integration with streaming
  - Event type analysis and item type counting
  - User-friendly progress updates and formatting
- **Learning Outcomes**:
  - Master higher-level streaming patterns
  - Understand tool integration with streaming
  - Learn event type analysis and monitoring
  - Build user-friendly streaming interfaces

### `04_streaming_with_handoffs.py`
- **Core Focus**: Advanced streaming with agent handoffs and real-time transitions
- **Key Features**:
  - Multi-agent streaming with handoff capabilities
  - Real-time agent transition monitoring
  - Complex event handling (handoffs, tool calls, messages)
  - Specialized agent coordination and routing
- **Learning Outcomes**:
  - Master streaming with agent handoffs
  - Understand real-time agent transitions
  - Learn complex event processing patterns
  - Build sophisticated multi-agent streaming systems

---

## 🔍 Streaming Event Types Deep Dive

The module covers different levels of streaming events:

### Basic Events
- **`raw_response_event`**: Token-by-token text streaming
- **`ResponseTextDeltaEvent`**: Individual text deltas

### Higher-Level Events
- **`agent_updated_stream_event`**: Agent transitions and updates
- **`run_item_stream_event`**: Tool calls, messages, and outputs

### Advanced Events
- **`handoff_call_item`**: Agent handoff initiation
- **`handoff_output_item`**: Handoff completion and transition
- **`tool_call_item`**: Tool execution requests
- **`tool_call_output_item`**: Tool execution results
- **`message_output_item`**: Agent message outputs

---

## 🛠️ Configuration

- **API Keys:**  
  Place your Gemini API key in the root `.env` file:
  ```
  GEMINI_API_KEY=...
  ```

- **Dependencies:**  
  Managed at the project root. Use `uv sync` to install.

---

## 🧭 Learning Goals

- Master basic streaming patterns and real-time text output
- Understand token-by-token streaming with raw response events
- Learn higher-level streaming with run item events and tool integration
- Master streaming with agent handoffs and real-time transitions
- Build user-friendly streaming interfaces and progress monitoring
- Understand different streaming event types and their use cases
- Develop sophisticated multi-agent streaming systems
- Learn to analyze and debug streaming events effectively

---

## 🔗 References & Further Reading

- [OpenAI Agents SDK Streaming Documentation](https://openai.github.io/openai-agents-python/streaming/)
- [OpenAI Agents SDK Handoffs Documentation](https://openai.github.io/openai-agents-python/handoffs/)
- [Project Root README](../../README.md) for setup, philosophy, and structure

---

## 🗃️ Related Projects & Showcases

- **[02_runner Module](../02_runner/)**: Basic agent execution and RunResult exploration
- **[01_agents Module](../01_agents/)**: Foundation agent patterns and handoff basics
- **[Projects Folder](../projects/)**: Real-world agentic applications with streaming

---

**Learning by streaming, one event at a time!**
