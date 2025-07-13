"""
08_async_vs_sync.py

Description:
    Demonstrates advanced usage of synchronous vs asynchronous agent execution, error handling, and performance benchmarking
    using the OpenAI Agents SDK with the Litellm (Mistral) model. Includes examples of sync/async tool calls, concurrent
    agent runs, error handling, and performance comparisons.

Features:
    - Synchronous and asynchronous agent execution
    - Function tools with sync and async behavior
    - Concurrent agent runs with asyncio.gather
    - Robust error handling for both sync and async flows
    - Performance benchmarking: sync vs async sequential vs async concurrent
    - Professional output formatting for clarity in terminal

Environment variables:
    - MISTRAL_API_KEY: Required for Litellm API access.

Author: Zohaib Khan
Reference: https://github.com/openai/openai-agents-sdk
"""
# =========================
# Imports & Setup
# =========================
import asyncio
import os
import time
import threading
from typing import Any
from dotenv import load_dotenv, find_dotenv
from agents.extensions.models.litellm_model import LitellmModel
from agents import (
    Agent,
    RunResult,
    Runner,
    set_tracing_disabled,
    function_tool,
)

# =========================
# Environment & Model Setup
# =========================
load_dotenv(find_dotenv())
set_tracing_disabled(True)

MISTRAL_API_KEY: str | None = os.getenv("MISTRAL_API_KEY")
MODEL_NAME: str = "mistral/mistral-small-2506"

if not MISTRAL_API_KEY:
    print("❌ MISTRAL_API_KEY environment variable is required but not found.")
    raise ValueError("MISTRAL_API_KEY environment variable is required but not found.")

print(f"🚀 Initializing OpenRouter client with model: {MODEL_NAME}")
model = LitellmModel(model=MODEL_NAME, api_key=MISTRAL_API_KEY)
print(f"✅ Model configured successfully\n")

# =========================
# Function Tools
# =========================

@function_tool
def simulate_sync_math(n: int = 2) -> str:
    """
    Simulates a blocking synchronous math operation (n^2) with a delay.

    Args:
        n (int): The number to square. Defaults to 2.

    Returns:
        str: Result string with the squared value.
    """
    print(f"🔒 [simulate_sync_math] Blocking for {n} seconds...")
    time.sleep(n)
    result: int = n * n
    return f"✅ Sync: {n}^2 = {result}"

@function_tool
async def simulate_async_api_call(n: int = 2) -> str:
    """
    Simulates an asynchronous API call with a delay.

    Args:
        n (int): Number of seconds to sleep. Defaults to 2.

    Returns:
        str: Result string indicating async completion.
    """
    print(f"⚡ [simulate_async_api_call] Sleeping for {n} seconds...")
    await asyncio.sleep(n)
    return f"✅ Async API call simulated for {n} seconds"

@function_tool
def get_coroutine_info() -> str:
    """
    Returns information about the current thread and coroutine context.

    Returns:
        str: Thread and coroutine info.
    """
    thread_name = threading.current_thread().name
    try:
        task = asyncio.current_task()
        task_name = task.get_name() if task else 'N/A'
    except Exception:
        task_name = 'N/A'
    return f"🧵 Running in thread: {thread_name}, Coroutine: {task_name}"

@function_tool
def unstable_tool() -> str:
    """
    Tool that always raises a RuntimeError to demonstrate error handling.

    Raises:
        RuntimeError: Always raised to simulate failure.
    """
    print(f"⚡ [unstable_tool] Raising exception...")
    raise RuntimeError("💥 Random failure occurred!")

# =========================
# Demo Functions
# =========================
def sync_execution():
    """
    Demonstrates synchronous agent execution with tool calls.
    """
    print("\n" + "=" * 60)
    print("🔒 SYNC EXECUTION DEMO")
    print("=" * 60)
    agent: Agent = Agent(
        name="Math Agent",
        instructions="You are a slow math assistant. Use simulate_sync_math for math related queries and any math question. Also use get_coroutine_info for environment info.",
        tools=[simulate_sync_math, get_coroutine_info],
        model=model,
    )
    start_time = time.time()
    print("Starting sync execution...")
    try:
        result: RunResult = Runner.run_sync(
            agent, "Square 4 and tell me about the environment"
        )
        print(f"🧠 Sync result: {result.final_output}\n")
    except Exception as e:
        print(f"❌ Sync execution failed: {type(e).__name__} → {e}")
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Sync execution took: {elapsed:.2f} seconds\n")

def sync_error_handling():
    """
    Demonstrates error handling for synchronous agent execution.
    """
    print("\n" + "=" * 60)
    print("🔒 SYNC ERROR HANDLING DEMO")
    print("=" * 60)
    agent: Agent = Agent(
        name="Buggy Agent",
        instructions="Use the unstable tool and return whatever happens.",
        tools=[unstable_tool],
        model=model,
    )
    try:
        Runner.run_sync(agent, "Try using the unstable tool")
    except Exception as e:
        print(f"❌ Caught sync error: {type(e).__name__} → {e}")
    print()

def sync_performance_test() -> float:
    """
    Measures performance of sequential sync agent calls.

    Returns:
        float: Total time taken for sync calls.
    """
    print("\n" + "=" * 60)
    print("🔒 SEQUENTIAL SYNC PERFORMANCE TEST")
    print("=" * 60)
    agent = Agent(
        name="Perf Agent",
        instructions="Use the simulate_async_api_call tool.",
        tools=[simulate_async_api_call],
        model=model,
    )
    start = time.time()
    for i in range(3):
        try:
            Runner.run_sync(agent, f"Call the simulate_async_api_call tool for {i+1}")
        except Exception as e:
            print(f"❌ Sync call {i+1} failed: {type(e).__name__} → {e}")
    sync_time = time.time() - start
    print(f"⏱️ Total time (sync): {sync_time:.2f} seconds")
    return sync_time

# =========================
# Async Demo Functions
# =========================
async def async_execution():
    """
    Demonstrates asynchronous agent execution with tool calls.
    """
    print("\n" + "=" * 60)
    print("⚡ ASYNC EXECUTION DEMO")
    print("=" * 60)
    agent: Agent = Agent(
        name="Math Agent",
        instructions="You are a slow math assistant. Use simulate_sync_math for math related queries and any math question. Also use get_coroutine_info for environment info.",
        tools=[simulate_sync_math, get_coroutine_info],
        model=model,
    )
    start_time = time.time()
    print("Starting async execution...")
    try:
        result: RunResult = await Runner.run(
            agent, "Square 4 and tell me about the environment"
        )
        print(f"⚡ Async result: {result.final_output}\n")
    except Exception as e:
        print(f"❌ Async execution failed: {type(e).__name__} → {e}")
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Async execution took: {elapsed:.2f} seconds\n")

async def concurrent_agents():
    """
    Demonstrates running multiple agents concurrently with asyncio.gather.
    """
    print("\n" + "=" * 60)
    print("⚡ CONCURRENT ASYNC AGENTS DEMO")
    print("=" * 60)
    agent1: Agent = Agent(
        name="API 1",
        instructions="Do async work",
        tools=[simulate_async_api_call],
        model=model,
    )
    agent2: Agent = Agent(
        name="API 2",
        instructions="Do async work",
        tools=[simulate_async_api_call],
        model=model,
    )
    agent3: Agent = Agent(
        name="API 3",
        instructions="Do async work",
        tools=[simulate_async_api_call],
        model=model,
    )
    start_time = time.time()
    try:
        results = await asyncio.gather(
            Runner.run(agent1, "Call simulate_async_api_call"),
            Runner.run(agent2, "Call simulate_async_api_call"),
            Runner.run(agent3, "Call simulate_async_api_call"),
        )
        print("🚀 Results:")
        for i, result in enumerate(results, 1):
            print(f"Agent {i} → {getattr(result, 'final_output', result)}")
    except Exception as e:
        print(f"❌ Concurrent agents failed: {type(e).__name__} → {e}")
    end_time = time.time()
    print(f"🕒 Concurrent execution took: {end_time - start_time:.2f} seconds\n")

async def async_error_handling():
    """
    Demonstrates error handling for asynchronous agent execution.
    """
    print("\n" + "=" * 60)
    print("⚡ ASYNC ERROR HANDLING DEMO")
    print("=" * 60)
    agent: Agent = Agent(
        name="Buggy Agent",
        instructions="Use the unstable tool and return whatever happens.",
        tools=[unstable_tool],
        model=model,
    )
    try:
        await Runner.run(agent, "Try using the unstable tool")
    except Exception as e:
        print(f"❌ Caught async error: {type(e).__name__} → {e}")
    print()

async def async_performance_test() -> tuple[float, float]:
    """
    Measures performance of sequential and concurrent async agent calls.

    Returns:
        tuple[float, float]: (sequential_time, concurrent_time)
    """
    print("\n" + "=" * 60)
    print("⚡ SEQUENTIAL ASYNC PERFORMANCE TEST")
    print("=" * 60)
    agent = Agent(
        name="Perf Agent",
        instructions="Use the simulate_async_api_call tool.",
        tools=[simulate_async_api_call],
        model=model,
    )
    start = time.time()
    for i in range(3):
        try:
            await Runner.run(agent, f"Call the simulate_async_api_call tool for {i+1}")
        except Exception as e:
            print(f"❌ Async call {i+1} failed: {type(e).__name__} → {e}")
    seq_time = time.time() - start
    print(f"⏱️ Total time (async sequential): {seq_time:.2f} seconds")
    print("\n" + "=" * 60)
    print("⚡⚡ CONCURRENT ASYNC PERFORMANCE TEST")
    print("=" * 60)
    start = time.time()
    try:
        await asyncio.gather(*[
            Runner.run(agent, f"Call the simulate_async_api_call tool for {i+1}")
            for i in range(3)
        ])
    except Exception as e:
        print(f"❌ Concurrent async calls failed: {type(e).__name__} → {e}")
    conc_time = time.time() - start
    print(f"⏱️ Total time (async concurrent): {conc_time:.2f} seconds")
    return seq_time, conc_time

# =========================
# Main Entrypoint
# =========================
def main():
    """
    Main entrypoint for running all sync and async agent demos and performance tests.
    """
    print("\n🔁 OpenAI SDK - Custom Async vs Sync Mastery\n" + "=" * 60)
    sync_execution()
    sync_error_handling()
    sync_time = sync_performance_test()
    async def run_async_funcs():
        await async_execution()
        await concurrent_agents()
        await async_error_handling()
        async_seq_time, async_conc_time = await async_performance_test()
        print(f"\n🚀 Speedup: {sync_time / async_conc_time:.2f}x faster than sync (concurrent)")
        print(f"🚀 Async concurrent is {async_seq_time / async_conc_time:.2f}x faster than sequential async")
    asyncio.run(run_async_funcs())

if __name__ == "__main__":
    main()
