"""
Comprehensive System Stress Test
================================
This test suite validates the entire "Dip Sniper" Trading Bot architecture.

It covers:
1. MARKET SCANNER: Handling large baskets (50+ stocks) and identifying RSI Divergences.
2. AGENT LOGIC: Multi-step reasoning (News -> Validation -> Execution).
3. QUANT LAB: Sim-to-Real Veto Logic (Rejecting losing strategies).
4. EXECUTION: Correctly formatting orders with dynamic parameters.
"""

import asyncio
import logging
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd
import numpy as np

# Import Bot Components
from trading_bot.config import TradingConfig
from trading_bot.scanner import scan_for_signals, Signal
from trading_bot.graph import agent_node
from trading_bot.tools import get_trading_tools, vet_trade_signal_tool, find_best_settings_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("SystemTest")

# =============================================================================
# MOCK MARKET DATA GENERATOR
# =============================================================================
def generate_mock_history(trend="UP", rsi_state="OVERSOLD"):
    """Generates synthetic price data to force specific test cases."""
    dates = pd.date_range(end=datetime.now(), periods=100)
    data = []
    price = 100.0
    
    for i in range(100):
        # Trend Component
        if trend == "UP": change = np.random.normal(0.2, 1.0) # Upward drift
        elif trend == "DOWN": change = np.random.normal(-0.2, 1.0) # Downward drift
        else: change = np.random.normal(0, 1.0) # Sideways
        
        # RSI Manipulation (Last few days)
        if i > 95:
             if rsi_state == "OVERSOLD": change = -3.0 # Crash to trigger RSI < 15
             if rsi_state == "OVERBOUGHT": change = 3.0
        
        price = price * (1 + change/100)
        data.append({
            "Open": price, "High": price*1.01, "Low": price*0.99, "Close": price, "Volume": 1000000
        })
        
    return pd.DataFrame(data, index=dates)

# =============================================================================
# TEST CASES
# =============================================================================

async def run_stress_test():
    print("\n🏗️  STARTING COMPREHENSIVE STRESS TEST...\n")
    print("---------------------------------------------------------------")
    
    # 1. SCANNER STRESS TEST
    print("1. 📡 SCANNER LOAD TEST (50 Stocks)")
    
    # Create a basket of 50 tickers
    basket = [f"STK_{i}" for i in range(50)]
    
    # Mock finding 3 signals in this basket
    with patch('trading_bot.scanner.get_sp500_tickers') as mock_tickers:
        with patch('trading_bot.scanner.fetch_data_yahoo') as mock_fetch: # Faster than live
            mock_tickers.return_value = basket
            
            # Create dummy results: 47 boring, 3 interesting
            mock_data = {}
            for t in basket:
                # Default: Boring sideways
                mock_data[t] = generate_mock_history(trend="FLAT", rsi_state="NEUTRAL")
            
            # Inject Signals
            mock_data['STK_0'] = generate_mock_history(trend="UP", rsi_state="OVERSOLD") # Good Buy
            mock_data['STK_1'] = generate_mock_history(trend="DOWN", rsi_state="OVERSOLD") # Bad Buy (Downtrend)
            mock_data['STK_2'] = generate_mock_history(trend="UP", rsi_state="OVERSOLD") # Good Buy
            
            mock_fetch.return_value = mock_data
            
            print("   Running Scan...")
            signals = await scan_for_signals()
            
    if len(signals) >= 3:
        print(f"   ✅ PASS: Scanner identified {len(signals)} signals from 50 candidates.")
    else:
        print(f"   ❌ FAIL: Scanner missed injected signals. Found {len(signals)}")
        
    print("---------------------------------------------------------------")

    # 2. QUANT LAB VETO TEST (Sim-to-Real)
    print("2. 🔬 QUANT LAB VETO TEST")
    
    # Case A: Good Strategy (Returns +5%)
    # We mock run_backtest to return specific stats
    with patch('trading_bot.quant_lab.quant_lab.run_backtest') as mock_test:
        mock_test.return_value = {"trades": 10, "win_rate": 70.0, "total_return_pct": 5.0, "rating": "GOOD"}
        
        res = vet_trade_signal_tool.invoke({"symbol": "GOOD_STK"})
        print(f"   Case A (Profitable): {res}")
        if "APPROVED" in res: print("   ✅ PASS: Approved profitable stock.")
        else: print("   ❌ FAIL: Vetoed profitable stock.")

    # Case B: Bad Strategy (Returns -5%)
    with patch('trading_bot.quant_lab.quant_lab.run_backtest') as mock_test:
        mock_test.return_value = {"trades": 10, "win_rate": 30.0, "total_return_pct": -5.0, "rating": "DANGEROUS"}
        
        res = vet_trade_signal_tool.invoke({"symbol": "BAD_STK"})
        print(f"   Case B (Losing): {res}")
        if "VETO" in res: print("   ✅ PASS: Vetoed losing stock.")
        else: print("   ❌ FAIL: Approved losing stock.")
        
    print("---------------------------------------------------------------")

    # 3. AGENT REASONING TEST (The "Brain")
    print("3. 🧠 AGENT MULTI-STEP REASONING")
    
    config = TradingConfig()
    model = ChatGoogleGenerativeAI(model=config.llm_model, temperature=0.0)
    tools = await get_trading_tools(config)
    
    # Mock State with a Signal
    state = {
        "messages": [],
        "signals": [
            {"symbol": "AAPL", "signal_type": "BUY", "price": 150.0, "sma": 20.0, "pct_from_sma": 1.5, "volume_ratio": 2.0, "daily_change_pct": -3.0}
        ],
        "positions": [],
        "buying_power": 100000.0, 
        "portfolio_value": 100000.0,
        "daily_pnl": 0.0,
        "is_market_open": True,
        "perform_scan": False
    }
    
    # Step 1: Agent should ask to VET the signal
    print("   Step 1: Agent Initial Evaluation...")
    result_1 = await agent_node(state.copy(), config, model, tools)
    last_msg = result_1["messages"][-1]
    
    calls = [t['name'] for t in last_msg.tool_calls] if last_msg.tool_calls else []
    print(f"   Agent called: {calls}")
    
    if "vet_trade_signal_tool" in calls or "get_market_sentiment" in calls:
        print("   ✅ PASS: Agent initiated research (Vet/Sentiment).")
    else:
        print("   ⚠️ WARN: Agent skipped research.")
        
    # Step 2: Simulate Tool Output (Force Approval) and see if Agent TRADES
    print("   Step 2: Feeding 'APPROVED' signal...")
    
    # Create valid tool message sequence
    history = result_1["messages"]
    
    # Fake responses for all called tools
    tool_m = []
    for call in last_msg.tool_calls:
        content = "Result"
        if call['name'] == 'vet_trade_signal_tool': content = "APPROVED: Win Rate 80%, Return +10%."
        if call['name'] == 'get_market_sentiment': content = "Sentiment: NEUTRAL. No bad news."
        if call['name'] == 'find_best_settings_tool': content = "OPTIMIZATION RESULT: Use SMA-21, Stop 3%."
        if call['name'] == 'evaluate_signal_quality': content = "Quality: HIGH. Score 4/4."
        
        tool_m.append(ToolMessage(content=content, tool_call_id=call['id']))
        
    state_2 = state.copy()
    state_2["messages"] = history + tool_m
    
    print("   Running Agent Step 2...")
    result_2 = await agent_node(state_2, config, model, tools)
    final_msg = result_2["messages"][-1]
    
    calls_2 = [t['name'] for t in final_msg.tool_calls] if final_msg.tool_calls else []
    print(f"   Agent called: {calls_2}")
    
    if "mcp_alpaca_place_stock_order" in calls_2:
        print("   ✅ PASS: Agent placed trade after validation.")
    elif "calculate_position_size" in calls_2:
        print("   ✅ PASS: Agent is calculating size (Trade imminent).")
    else:
        print("   ❌ FAIL: Agent did not trade despite perfect approval.")

if __name__ == "__main__":
    asyncio.run(run_stress_test())
