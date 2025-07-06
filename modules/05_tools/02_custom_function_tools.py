"""
Custom Function Tools Demo - OpenAI Agents SDK

This module demonstrates advanced function tool usage with the OpenAI Agents SDK.
It showcases:
- Creating custom FunctionTool instances with manual configuration
- Using Pydantic models for structured tool parameters
- Async tool functions with context management
- Product inventory management system
- Tool introspection and metadata display
- Error handling and graceful failure

The demo implements a simple product inventory system with add and search capabilities.

Environment variables:
    - GEMINI_API_KEY: Required for Gemini model access.

Author: Zohaib Khan
"""

from openai import AsyncOpenAI
from agents import (
    Runner,
    Agent,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    RunResult,
    function_tool,
    RunContextWrapper,
    FunctionTool,
)
from dotenv import load_dotenv, find_dotenv
import os
from pydantic import BaseModel
from pprint import pprint
import asyncio

# =========================
# Environment & Model Setup
# =========================

load_dotenv(find_dotenv())
set_tracing_disabled(True)

# Configuration constants
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_NAME: str = "gemini-2.0-flash"

# Validate API key
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY environment variable is required but not found.")
    raise ValueError("GEMINI_API_KEY environment variable is required but not found.")

print(f"🚀 Initializing Gemini client with model: {GEMINI_MODEL_NAME}")

# Initialize external OpenAI client for Gemini
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL,
)

# Configure the chat completions model
model = OpenAIChatCompletionsModel(
    openai_client=external_client, model=GEMINI_MODEL_NAME
)

print(f"✅ Model configured successfully\n")


# =========================
# Data Models
# =========================

class NewProduct(BaseModel):
    """Pydantic model for new product data."""
    name: str
    price: float
    category: str


class ProductSearch(BaseModel):
    """Pydantic model for product search queries."""
    category: str
    max_results: int


# =========================
# Mock Database
# =========================

mock_db = []


# =========================
# Custom Tool Functions
# =========================

async def add_product(ctx: RunContextWrapper, args: str) -> str:
    """
    Add a new product to the inventory.
    
    Args:
        ctx: RunContextWrapper for managing execution context
        args: JSON string containing product data
        
    Returns:
        str: Confirmation message with product details
    """
    try:
        data = NewProduct.model_validate_json(args)
        mock_db.append(data)
        return f"✅ Product '{data.name}' added under category '{data.category}' with price ${data.price}"
    except Exception as e:
        return f"❌ Error adding product: {str(e)}"


async def search_products(ctx: RunContextWrapper, args: str) -> str:
    """
    Search for products by category with optional result limit.
    
    Args:
        ctx: RunContextWrapper for managing execution context
        args: JSON string containing search parameters
        
    Returns:
        str: Search results or error message
    """
    try:
        data = ProductSearch.model_validate_json(args)
        results = [p for p in mock_db if p.category == data.category][: data.max_results]
        return f"🔍 Found {len(results)} products in category '{data.category}':\n{results}"
    except Exception as e:
        return f"❌ Error searching products: {str(e)}"


# =========================
# Tool Configuration
# =========================

add_product_tool: FunctionTool = FunctionTool(
    name="add_product",
    description="Add a new product to the inventory",
    params_json_schema=NewProduct.model_json_schema(),
    on_invoke_tool=add_product,
)

search_product_tool: FunctionTool = FunctionTool(
    name="search_products",
    description="Search for products by category",
    params_json_schema=ProductSearch.model_json_schema(),
    on_invoke_tool=search_products,
)

# =========================
# Agent Configuration
# =========================

agent = Agent(
    name="Product Assistant",
    instructions="""You are an assistant that manages product inventory.
    You can:
    1. Add new products with name, price, and category
    2. Search products by category
    """,
    tools=[add_product_tool, search_product_tool],
    model=model,
)


# =========================
# Main Execution Function
# =========================

async def main():
    """
    Execute the product inventory management demo.
    
    Demonstrates adding products and searching the inventory.
    """
    print("🛍️ Product Inventory Management Demo")
    print("=" * 50)
    
    try:
        # Add first product
        print("\n📦 Adding First Product...")
        result1 = await Runner.run(
            agent,
            input="Add a product named 'Laptop', priced at 1200, in category 'Electronics'",
        )
        print(f"Result: {result1.final_output}")

        # Add second product
        print("\n📦 Adding Second Product...")
        result2 = await Runner.run(
            agent, input="Add a product 'Mouse', 25 dollars, in 'Electronics'"
        )
        print(f"Result: {result2.final_output}")

        # Search products
        print("\n🔍 Searching Products...")
        result3 = await Runner.run(
            agent, input="Show me up to 5 products in Electronics category"
        )
        print(f"Result: {result3.final_output}")
        
        # Display tool metadata
        print("\n🔧 Tool Metadata:")
        for tool in agent.tools:
            print(f"  - {tool.name}: {tool.description}")
            
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 