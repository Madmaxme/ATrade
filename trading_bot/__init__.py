"""
Trading Bot
============
A LangGraph-powered autonomous day trading agent.
"""

from trading_bot.config import TradingConfig
from trading_bot.graph import create_trading_graph, TradingState
from trading_bot.scanner import scan_for_signals, Signal
from trading_bot.scheduler import TradingScheduler

__all__ = [
    "TradingConfig",
    "create_trading_graph",
    "TradingState",
    "scan_for_signals",
    "Signal",
    "TradingScheduler",
]

__version__ = "0.1.0"
