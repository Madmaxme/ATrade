
import asyncio
import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.tools import tool

# Load env
load_dotenv()

from trading_bot.config import DEFAULT_CONFIG

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

async def test_llm():
    model_name = DEFAULT_CONFIG.llm_model
    region = DEFAULT_CONFIG.aws_region
    print(f"Testing model: {model_name} in region {region}...")
    
    try:
        llm = ChatBedrock(
            model_id=model_name,
            region_name=region,
            model_kwargs={"temperature": 0}
        )
        
        llm_with_tools = llm.bind_tools([multiply])
        
        print("Invoking model with a tool call request...")
        result = await llm_with_tools.ainvoke("What is 5 times 8?")
        
        print(f"Result type: {type(result)}")
        print(f"Tool calls: {result.tool_calls}")
        
        if result.tool_calls and result.tool_calls[0]["name"] == "multiply":
            print("✅ SUCCESS: Model successfully generated a tool call!")
        else:
            print("⚠️ WARNING: Model responded but did not call the tool correctly.")
            print(f"Content: {result.content}")

    except Exception as e:
        print(f"❌ TEST FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test_llm())
