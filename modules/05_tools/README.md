# 🔧 05_tools — Function Tools & Advanced Tool Integration

_A comprehensive exploration of function tools in the OpenAI Agents SDK, covering basic tools, custom configurations, agents as tools, error handling, context-aware tools, and advanced tool choice patterns._

---

## 📚 About This Module

This folder demonstrates the full spectrum of function tool capabilities in the OpenAI Agents SDK. Here, I explore everything from basic function tools to advanced patterns like agents-as-tools, context-aware tools, error handling, and sophisticated tool choice configurations. Each script showcases practical tool integration scenarios with detailed examples and real-world applications.

---

## 🗂️ Contents

| File/Notebook                   | Description                                                              |
| ------------------------------- | ------------------------------------------------------------------------ |
| `01_basic_function_tools.py`    | Basic function tools with @function_tool decorator and Pydantic models.  |
| `02_custom_function_tools.py`   | Custom FunctionTool instances with manual configuration and async tools. |
| `03_advanced_function_tools.py` | Advanced tools with strict parameter validation and ConfigDict.          |
| `04_agents_as_tools.py`         | Using agents as tools within other agents for complex workflows.         |
| `05_error_handling_tools.py`    | Advanced error handling strategies and custom error handlers.            |
| `06_context_aware_tools.py`     | Context-aware tools with user sessions and personalized responses.       |
| `07_tool_choice_streaming.py`   | Advanced tool choice configurations and streaming with tools.            |

---

## 🏁 Quick Start

> **Prerequisites:**
>
> - Python environment with dependencies installed (`uv sync`)
> - `.env` file with your API key(s) in the project root

### Run Basic Tool Examples

```bash
# Basic function tools with decorator
uv run modules/05_tools/01_basic_function_tools.py

# Custom function tools with manual configuration
uv run modules/05_tools/02_custom_function_tools.py

# Advanced tools with strict validation
uv run modules/05_tools/03_advanced_function_tools.py
```

### Run Advanced Tool Examples

```bash
# Agents as tools for complex workflows
uv run modules/05_tools/04_agents_as_tools.py

# Error handling tools with custom handlers
uv run modules/05_tools/05_error_handling_tools.py

# Context-aware tools with user sessions
uv run modules/05_tools/06_context_aware_tools.py

# Tool choice configurations and streaming
uv run modules/05_tools/07_tool_choice_streaming.py
```

---

## 📄 File Summaries

### `01_basic_function_tools.py`

- **Core Focus**: Basic function tools with @function_tool decorator
- **Key Features**:
  - Simple function tools using @function_tool decorator
  - Pydantic models for structured parameters
  - University database simulation
  - Tool introspection and metadata display
  - Context management with RunContextWrapper
- **Learning Outcomes**:
  - Understand basic function tool creation
  - Learn to use Pydantic models for parameters
  - Master tool introspection and metadata
  - Build confidence in basic tool integration

### `02_custom_function_tools.py`

- **Core Focus**: Custom FunctionTool instances with manual configuration
- **Key Features**:
  - Manual FunctionTool configuration
  - Async tool functions with context management
  - Product inventory management system
  - Custom parameter schemas
  - Advanced tool introspection
- **Learning Outcomes**:
  - Master custom FunctionTool creation
  - Understand async tool functions
  - Learn manual tool configuration
  - Build complex tool systems

### `03_advanced_function_tools.py`

- **Core Focus**: Advanced tools with strict parameter validation
- **Key Features**:
  - Strict parameter validation with ConfigDict
  - "forbid" extra fields configuration
  - Advanced error handling patterns
  - User data processing system
  - Strict schema validation
- **Learning Outcomes**:
  - Master strict parameter validation
  - Understand ConfigDict configurations
  - Learn advanced error handling
  - Build robust tool systems

### `04_agents_as_tools.py`

- **Core Focus**: Using agents as tools within other agents
- **Key Features**:
  - Specialized agents for different tasks
  - Converting agents to tools with as_tool()
  - Multi-agent orchestration patterns
  - Content creation and editing workflows
  - Complex agent collaboration
- **Learning Outcomes**:
  - Master agents-as-tools patterns
  - Understand multi-agent orchestration
  - Learn complex workflow design
  - Build sophisticated agent systems

### `05_error_handling_tools.py`

- **Core Focus**: Advanced error handling strategies and custom error handlers
- **Key Features**:
  - Custom error handlers for function tools
  - Different error handling strategies
  - Streaming execution with error events
  - Tool failure recovery and user guidance
  - Error type detection and appropriate responses
- **Learning Outcomes**:
  - Master custom error handling
  - Understand different error strategies
  - Learn tool failure recovery
  - Build robust error-resistant systems

### `06_context_aware_tools.py`

- **Core Focus**: Context-aware tools with user sessions and personalization
- **Key Features**:
  - User session context management
  - Personalized tool responses
  - Usage statistics tracking
  - Multilingual support with preferences
  - Chain multiple tools for complex workflows
- **Learning Outcomes**:
  - Master context-aware tool design
  - Understand user session management
  - Learn personalized tool responses
  - Build sophisticated user-aware systems

### `07_tool_choice_streaming.py`

- **Core Focus**: Advanced tool choice configurations and streaming with tools
- **Key Features**:
  - ModelSettings with different tool_choice options
  - Parallel vs sequential tool calls
  - Streaming tool execution with event handling
  - Study productivity tools with focus scoring
  - Advanced tool choice patterns
- **Learning Outcomes**:
  - Master tool choice configurations
  - Understand streaming with tools
  - Learn parallel vs sequential execution
  - Build advanced tool orchestration systems

---

## 🔍 Tool Patterns Deep Dive

The module covers different levels of tool complexity:

### Basic Tool Patterns

- **@function_tool decorator**: Simple function tool creation
- **Pydantic models**: Structured parameter validation
- **Tool introspection**: Metadata and schema inspection

### Advanced Tool Patterns

- **Custom FunctionTool**: Manual tool configuration
- **Strict validation**: ConfigDict with "forbid" extra fields
- **Async tools**: Asynchronous function tools

### Complex Tool Patterns

- **Agents as tools**: Multi-agent orchestration
- **Context-aware tools**: User session management
- **Error handling**: Custom error handlers and recovery
- **Tool choice**: Advanced configuration patterns

---

## 🛠️ Configuration

- **API Keys:**  
  Place your API keys in the root `.env` file:

  ```
  GEMINI_API_KEY=...
  MISTRAL_API_KEY=...
  ```

- **Dependencies:**  
  Managed at the project root. Use `uv sync` to install.

---

## 🧭 Learning Goals

- Master basic function tool creation with @function_tool decorator
- Understand custom FunctionTool configuration and async tools
- Learn advanced parameter validation and strict schemas
- Master agents-as-tools patterns and multi-agent orchestration
- Build robust error handling strategies for tools
- Create context-aware tools with user session management
- Master advanced tool choice configurations and streaming
- Develop sophisticated tool orchestration and workflow systems
- Understand different tool patterns and their use cases
- Build production-ready tool integration systems

---

## 🔗 References & Further Reading

- [OpenAI Agents SDK Tools Documentation](https://openai.github.io/openai-agents-python/tools/)
- [OpenAI Agents SDK Handoffs Documentation](https://openai.github.io/openai-agents-python/handoffs/)
- [Project Root README](../../README.md) for setup, philosophy, and structure

---

## 🗃️ Related Projects & Showcases

- **[01_agents Module](../01_agents/)**: Foundation agent patterns and basic tool integration
- **[02_runner Module](../02_runner/)**: Agent execution and tool usage patterns
- **[04_stream Module](../04_stream/)**: Streaming with tools and real-time events
- **[Projects Folder](../projects/)**: Real-world agentic applications with advanced tool integration

---

**Learning by building tools, one function at a time!**
