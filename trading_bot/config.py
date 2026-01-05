"""
Trading Bot Configuration
==========================
All configurable parameters for the trading bot.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class TradingConfig:
    """Configuration for the trading bot."""
    
    def get_now_et(self):
        """Get current time in Eastern Timezone (robust)."""
        import pytz
        from datetime import datetime
        # Get UTC time first, then convert to ET
        # This handles containers running in UTC or other timezones correctly
        return datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern'))
    
    # =========================================================================
    # SYSTEM PATHS
    # =========================================================================
    
    # Directory for persistent data (memory, logs, journals)
    # Defaults to current directory if not set (e.g. locally)
    # On Railway, set this to /app/data
    @property
    def data_dir(self) -> str:
        d = os.getenv("DATA_DIR", ".")
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        return d
    
    # =========================================================================
    # RISK MANAGEMENT (CRITICAL - DON'T CHANGE WITHOUT UNDERSTANDING)
    # =========================================================================
    
    # Maximum percentage of portfolio per trade (e.g., 0.20 = 20%)
    # Increased for "Sniper Strategy" since we only take 1 trade per day
    max_position_size_pct: float = 0.20
    
    # Maximum number of concurrent positions
    # STRICT LIMIT: One stock per day ("Daily Champion")
    max_positions: int = 1
    
    # Stop loss percentage per trade (e.g., 0.01 = 1%)
    stop_loss_pct: float = 0.01
    
    # Take profit percentage per trade (e.g., 0.02 = 2%, gives 2:1 reward/risk)
    take_profit_pct: float = 0.02
    
    # Maximum daily loss before stopping (e.g., 0.02 = 2% of portfolio)
    max_daily_loss_pct: float = 0.02
    
    # Time to close all positions (24h format, ET)
    close_positions_time: str = "15:55"
    
    # =========================================================================
    # STRATEGY PARAMETERS
    # =========================================================================
    
    # SMA period for crossover detection
    sma_period: int = 21
    
    # Minimum volume ratio to consider a signal valid
    min_volume_ratio: float = 0.5
    
    # Maximum volume ratio (avoid extreme spikes that might reverse)
    max_volume_ratio: float = 5.0
    
    # Minimum price for stocks to trade
    min_stock_price: float = 10.0
    
    # Maximum price for stocks to trade
    max_stock_price: float = 500.0
    
    # =========================================================================
    # SCHEDULE
    # =========================================================================
    
    # Market hours (ET)
    market_open: str = "09:30"
    market_close: str = "16:00"
    
    # How often to scan for new signals (minutes)
    scan_interval_minutes: int = 5
    
    # How often to check position status (seconds)
    position_check_interval_seconds: int = 30
    
    # =========================================================================
    # FILTERS
    # =========================================================================
    
    # Skip stocks with earnings in next N days
    earnings_blackout_days: int = 7
    
    # Only trade stocks in these sectors (empty = all)
    allowed_sectors: List[str] = field(default_factory=list)
    
    # Never trade these tickers
    blacklist_tickers: List[str] = field(default_factory=lambda: [
        "GME", "AMC", "BBBY",  # Meme stocks - too volatile
    ])
    
    # =========================================================================
    # ALPACA SETTINGS
    # =========================================================================
    
    # Use paper trading (set to False for live - BE CAREFUL)
    paper_trading: bool = True
    
    # Alpaca API credentials (loaded from environment)
    @property
    def alpaca_api_key(self) -> str:
        return os.getenv("ALPACA_API_KEY", "")
    
    @property
    def alpaca_secret_key(self) -> str:
        return os.getenv("ALPACA_SECRET_KEY", "")
    
    # =========================================================================
    # LLM SETTINGS
    # =========================================================================
    
    # Model to use for trading decisions
    llm_model: str = "gemini-3-flash-preview"
    
    # Temperature for LLM (lower = more consistent)
    llm_temperature: float = 0.1
    
    # =========================================================================
    # LOGGING
    # =========================================================================
    
    # Log level
    log_level: str = "INFO"
    
    # Save trade logs to file
    save_trade_logs: bool = True
    
    # Trade log directory
    log_directory: str = "logs"
    
    # =========================================================================
    # GRAPH SETTINGS
    # =========================================================================
    
    # Recursion limit for LangGraph execution (max steps per cycle)
    recursion_limit: int = 200


# Default configuration instance
DEFAULT_CONFIG = TradingConfig()
