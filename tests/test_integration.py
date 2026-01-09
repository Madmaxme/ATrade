"""
Integration Test: Agent Flow
============================
Tests the full pipeline: Scanner -> Agent -> Quant Lab Tools.
Ensures the Agent *autonomously* decides to use the new Validation/Optimization tools
when presented with a trading signal.
"""

import asyncio
import logging
from unittest.mock import patch
from datetime import datetime

# Import Agent Components
from trading_bot.config import TradingConfig
from trading_bot.scanner import scan_for_signals, Signal
from trading_bot.graph import agent_node
from trading_bot.tools import get_trading_tools
from langchain_google_genai import ChatGoogleGenerativeAI

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("IntegrationTest")

async def run_integration_test():
    print("\n🤖 STARTING AGENT FLOW INTEGRATION TEST...\n")
    
    # 1. Setup Configuration
    config = TradingConfig()
    
    # 2. Mock the Scanner to avoid analysing 500 stocks (Speed)
    # We force it to look at a few distinct profiles:
    # - Diverse mix (Tech, Pharma, Banks, Retail, Energy)
    print("1. 📡 Running Scanner (Scanning 20 Diverse Stocks)...")
    
    # Mocking get_sp500_tickers to return a diverse 20-stock basket
    basket = [
        'NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', # Tech/Growth
        'JPM', 'BAC', 'GS', # Finance
        'PFE', 'MRK', 'LLY', # Pharma
        'XOM', 'CVX', # Energy
        'WMT', 'TGT', # Retail
        'F', 'GM' # Auto
    ]
    
    with patch('trading_bot.scanner.get_sp500_tickers') as mock_tickers:
        mock_tickers.return_value = basket
        
        # Run real scanner logic
        signals = await scan_for_signals()
        
    if not signals:
        print("   ⚠️ No live signals found today. Injecting a MOCK signal to test Agent logic.")
        # Inject a dummy signal so we can still test the Agent's reaction
        signals = [
            Signal(
                symbol="NVDA",
                signal_type="BUY",
                price=140.0,
                sma=135.0,
                pct_from_sma=3.7,
                volume_ratio=2.1,
                daily_change_pct=2.5,
                timestamp=datetime.now()
            )
        ]
    else:
        print(f"   ✅ Scanner found {len(signals)} actual market signals!")
        for s in signals:
            print(f"      - {s.symbol} ({s.signal_type}) Vol Ratio: {s.volume_ratio:.2f}")

    # 3. Setup Agent State
    print("\n2. 🧠 Initializing Agent with Signals...")
    state = {
        "messages": [],
        "signals": [s.__dict__ for s in signals], # Convert dataclass to dict
        "positions": [],
        "buying_power": 100000.0,
        "portfolio_value": 100000.0,
        "daily_pnl": 0.0,
        "is_market_open": True,
        "perform_scan": False # We already scanned
    }
    
    # 4. Initialize Tools & Model
    tools = await get_trading_tools(config)
    
    model = ChatGoogleGenerativeAI(
        model=config.llm_model,
        temperature=0.0 # Force determinism
    )
    
    # 5. Run Agent Node
    print("3. ⚡ Invoking Agent Node (Thinking)...")
    result = await agent_node(state, config, model, tools)
    
    # 6. Analyze Agent Decision
    last_msg = result["messages"][-1]
    
    print("\n4. 🧐 Analyzing Agent Response:")
    
    if last_msg.tool_calls:
        print(f"   Agent wants to call {len(last_msg.tool_calls)} tools:")
        found_veto = False
        found_opt = False
        
        for tool_call in last_msg.tool_calls:
            name = tool_call['name']
            args = tool_call['args']
            print(f"   👉 Tool: {name} | Args: {args}")
            
            if name == 'vet_trade_signal_tool':
                found_veto = True
            if name == 'find_best_settings_tool':
                found_opt = True
                
        if found_veto:
            print("\n   ✅ SUCCESS: Agent correctly attempted to VETO/VALIDATE the signal first.")
        else:
            print("\n   ⚠️ WARNING: Agent did not call 'vet_trade_signal_tool'. It might be rushing to trade or skipping.")
            
        if found_opt:
             print("   ✅ SUCCESS: Agent also attempted to OPTIMIZE parameters.")
             
        # 7. Execute the tools manually to show the user what would happen
        print("\n5. ⛓️ Simulating Tool Execution (Closing the Loop)...")
        from trading_bot.tools import vet_trade_signal_tool, find_best_settings_tool
        
        for tool_call in last_msg.tool_calls:
            name = tool_call['name']
            args = tool_call['args']
            
            if name == 'vet_trade_signal_tool':
                print(f"   Executing {name}...")
                res = vet_trade_signal_tool.invoke(args)
                print(f"   Output: {res}")
                
            elif name == 'find_best_settings_tool':
                 print(f"   Executing {name}...")
                 res = find_best_settings_tool.invoke(args)
                 print(f"   Output: {res}")

    else:
        print("   Agent returned text only (No tool calls).")
        print(f"   Content: {last_msg.content}")
        print("\n   ❌ FAILURE: Agent did not attempt to use any tools.")

if __name__ == "__main__":
    asyncio.run(run_integration_test())
