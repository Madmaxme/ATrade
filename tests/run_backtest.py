#!/usr/bin/env python3
"""
Run Backtest
============
Script to run the strategic backtester (Mode A).
Usage: python run_backtest.py
"""

import asyncio
import os
from dotenv import load_dotenv

# Load env vars first
load_dotenv()

from trading_bot.backtester import Backtester
from trading_bot.config import TradingConfig

async def main():
    print("="*60)
    print("  ATRADE STRATEGY BACKTESTER")
    print("  Testing: SMA + Wide Stops + Vol Filter (LIKELY TARGETS UNIVERSE)")
    print("="*60)
    
    # 1. Setup Config
    config = TradingConfig()
    
    # Override defaults for testing if desired
    # config.sma_period = 21
    # Note: We are now using the optimized defaults in config.py
    
    # 2. Select Tickers: "Likely Targets"
    # These are the highest volume/liquidity names the Agent is most likely to surface
    # when sorting by volume metrics.
    test_universe = [
        # Magnificent 7 / Tech Leaders
        'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX',
        # High Volume / Retail Favorites
        'PLTR', 'SOFI', 'MARA', 'COIN', 'HOOD', 'ROKU', 'DKNG',
        # Financials / Banks
        'JPM', 'BAC', 'WFC', 'C', 'V', 'MA',
        # Industrial / Energy / Retail
        'XOM', 'CVX', 'BA', 'GE', 'WMT', 'COST', 'TGT',
        # Semis
        'INTC', 'MU', 'QCOM', 'AVGO'
    ]
    print(f"   ℹ️  Testing 'Likely Targets' Universe ({len(test_universe)} High-Activity Stocks)...")

    # 3. Create Runner
    backtester = Backtester(config=config, initial_capital=10000)
    
    # 4. Fetch Data (e.g., last 365 days)
    await backtester.fetch_data(test_universe, days_back=365)
    
    # 5. Run
    backtester.run()

if __name__ == "__main__":
    asyncio.run(main())
