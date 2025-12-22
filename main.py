#!/usr/bin/env python3
"""
Day Trading Bot - Main Entry Point
===================================
A LangGraph-powered autonomous day trading agent using Alpaca MCP for execution.

This bot:
1. Scans for SMA crossover opportunities pre-market
2. Enters positions when signals trigger
3. Manages positions throughout the day (stops, targets)
4. Closes all positions before market close (day trading)
5. Logs everything for analysis
"""

import asyncio
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

from trading_bot.graph import create_trading_graph
from trading_bot.scheduler import TradingScheduler
from trading_bot.config import TradingConfig

# Configure logging to suppress noisy output
logging.basicConfig(level=logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

load_dotenv()


async def main():
    """Main entry point for the trading bot."""
    
    print("=" * 60)
    print("  ATRADE - Autonomous Trading Agent")
    print("  Powered by LangGraph + Alpaca MCP")
    print("=" * 60)
    print(f"\n  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    config = TradingConfig()
    
    # Validate environment
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not set")
    if not os.getenv("ALPACA_API_KEY"):
        raise ValueError("ALPACA_API_KEY not set")
    if not os.getenv("ALPACA_SECRET_KEY"):
        raise ValueError("ALPACA_SECRET_KEY not set")
    
    print(f"\n  Status: {'🟢 PAPER TRADING (Safe Mode)' if config.paper_trading else '⚠️  LIVE TRADING (Real Money)'}")
    print(f"  Configuration: Max {config.max_positions} positions | Risk {config.max_daily_loss_pct}% daily")
    print("\n" + "=" * 60)
    
    # Create the trading graph
    graph = await create_trading_graph(config)
    
    # Create and start the scheduler
    scheduler = TradingScheduler(graph, config)
    
    try:
        await scheduler.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Shutdown requested...")
        await scheduler.shutdown()
        print("\n✅ Bot stopped safely. Have a nice day!")


if __name__ == "__main__":
    asyncio.run(main())
