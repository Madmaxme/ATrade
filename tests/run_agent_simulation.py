#!/usr/bin/env python3
"""
Run Agent Simulation
====================
Script to run the Agent Simulator (Mode B).
"""

import asyncio
from dotenv import load_dotenv

# Load env vars first
load_dotenv()

from trading_bot.config import TradingConfig
from trading_bot.agent_simulator import AgentSimulator

async def main():
    print("="*60)
    print("  ATRADE AGENT REPLAY (Mode B)")
    print("  Testing: AI Psychology vs Historical Data")
    print("="*60)
    
    config = TradingConfig()
    
    # We only test on a few stocks to save tokens/time
    test_universe = ['AAPL', 'TSLA'] # Highly liquid, volatile
    
    sim = AgentSimulator(config, initial_capital=50000)
    
    print("   ⏳ Loading 7 days of history...")
    await sim.setup(test_universe, days_back=7)
    
    print("   🎮 Starting Simulation Loop...")
    await sim.run()

if __name__ == "__main__":
    asyncio.run(main())
