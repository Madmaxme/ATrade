"""
Full Agent Simulation (Market Closed)
=====================================
This script replicates 'main.py' but mocks the Market & Time conditions.
It forces the Agent to wake up, see a "Buy" signal, validate it, and execute it,
even though the real market is closed.

It is the closest thing to a "Full Dress Rehearsal" before tomorrow morning.
"""

import asyncio
import logging
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Import Real Bot Components
from trading_bot.config import TradingConfig
from trading_bot.graph import create_trading_graph
from trading_bot.scanner import Signal

# Use LangChain's built-in mocking if needed, but we start simple
logging.basicConfig(level=logging.INFO, format='%(message)s')

async def run_full_simulation():
    print("\n🎬 STARTING FULL DRESS REHEARSAL (SIMULATION)...\n")
    
    # 1. Mock Configuration (Force Market Open logic)
    config = TradingConfig()
    
    # 2. Mock The "Environment" (Scanner, Account, Time)
    print("1. 🎭 Setting the Stage (Mocking Market Conditions)...")
    
    # Mock Signals (What the scanner "Found")
    mock_signals = [
        Signal(
            symbol="AAPL", 
            signal_type="BUY", 
            price=145.0, 
            sma=25.0, # RSI Value in this field
            pct_from_sma=1.5, 
            volume_ratio=2.5, 
            daily_change_pct=-2.0, 
            timestamp=datetime.now()
        ),
         Signal(
            symbol="BAD_STK", 
            signal_type="BUY", 
            price=10.0, 
            sma=25.0, 
            pct_from_sma=-50.0, 
            volume_ratio=0.5, 
            daily_change_pct=-10.0, 
            timestamp=datetime.now()
        )
    ]
    
    # Mock Account (Buying Power)
    mock_account = MagicMock()
    mock_account.buying_power = "100000.00"
    mock_account.equity = "100000.00"
    mock_account.daytrade_count = 0
    
    # 3. Build the Real Graph
    print("2. 🤖 Waking up the Agent...")
    app = await create_trading_graph(config)
    
    # 4. Run the Graph with Injected State
    # We bypass the "Entry Node" (which checks time) and inject state directly
    # pretending the scanner already ran.
    
    initial_state = {
        "messages": [],
        "signals": [s.__dict__ for s in mock_signals], # Signals are ready
        "positions": [],
        "buying_power": 100000.0,
        "portfolio_value": 100000.0,
        "daily_pnl": 0.0,
        "is_market_open": True, # Force Open
        "perform_scan": False   # Skip scanning, we have signals
    }
    
    print("3. ▶️  ACTION! (Agent is thinking)...")
    
    # We Mock the "verify_market_open" inside the graph nodes if necessary
    # But since we pass 'is_market_open': True in state, the graph should respect it.
    
    config_dict = {"configurable": {"thread_id": "SIMULATION_1"}}
    
    # Create a loop to let the agent cycle multiple times (Think -> Act -> Result -> Think -> Act)
    # We limit to 5 steps to prevent infinite loops if it gets stuck
    step_count = 0
    async for event in app.astream(initial_state, config_dict):
        step_count += 1
        for key, value in event.items():
            print(f"\n   📍 Node: {key}")
            if "messages" in value and value["messages"]:
                last_msg = value["messages"][-1]
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    print(f"      🛠️  Tools Called: {[t['name'] for t in last_msg.tool_calls]}")
                elif hasattr(last_msg, 'content'):
                    # Truncate content for readability
                    content = last_msg.content[:100] + "..." if len(last_msg.content) > 100 else last_msg.content
                    print(f"      🗣️  Agent Said: {content}")
                    
        if step_count >= 8:
            print("\n   🛑 Cutting scene (Max steps reached).")
            break

if __name__ == "__main__":
    # We set environment variables or mock time if needed, but state injection should suffice
    asyncio.run(run_full_simulation())
