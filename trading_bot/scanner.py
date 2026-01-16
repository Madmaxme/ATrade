"""
Trading Signal Scanner
=======================
Scans S&P 500 stocks for SMA crossover signals.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass
import requests
from io import StringIO

# Try to import data sources
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    ALPACA_DATA_AVAILABLE = True
except ImportError:
    ALPACA_DATA_AVAILABLE = False


try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


@dataclass
class Signal:
    """Trading signal from the scanner."""
    symbol: str
    signal_type: str  # 'BUY' or 'SELL'
    price: float
    sma: float
    pct_from_sma: float
    volume_ratio: float
    daily_change_pct: float
    timestamp: datetime


# =============================================================================
# S&P 500 TICKERS
# =============================================================================

def get_sp500_tickers() -> List[str]:
    """Fetch S&P 500 tickers."""
def get_fallback_tickers() -> List[str]:
    """Return a static list of major S&P 500 tickers."""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
        'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'ABBV',
        'MRK', 'PFE', 'KO', 'PEP', 'COST', 'WMT', 'MCD', 'CSCO', 'CRM',
        'ADBE', 'NKE', 'ORCL', 'INTC', 'BA', 'CAT', 'IBM', 'GE', 'DIS'
    ]


def get_sp500_tickers() -> List[str]:
    """
    Fetch current S&P 500 ticker list.
    Falls back to a static list if web fetch fails.
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(StringIO(response.text))
        sp500_table = tables[0]
        # Clean ticker symbols (replace . with - for consistency, will be fixed for Alpaca later if needed)
        # Note: We will handle . vs - conversion in the scanner logic
        tickers = sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()
        print(f"   ✓ Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"   ⚠️ Could not fetch S&P 500 list: {e}")
        print("   ⚠️ Using static fallback list (~36 major stocks)")
        return get_fallback_tickers()


# =============================================================================
# SCANNER LOGIC
# =============================================================================

from trading_bot.config import TradingConfig, DEFAULT_CONFIG

# ... (dataclass and imports)

def calculate_rsi(prices: pd.Series, period: int = 2) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_rsi_signal(df: pd.DataFrame, rsi_col: str = 'RSI_2') -> Optional[str]:
    """Detect RSI Mean Reversion Signals."""
    if len(df) < 2: return None
    today = df.iloc[-1]
    if pd.isna(today[rsi_col]): return None
    if today[rsi_col] < 15: 
        return 'BUY'
    return None

def analyze_stock(ticker: str, df: pd.DataFrame, config: TradingConfig = DEFAULT_CONFIG) -> Optional[Signal]:
    """Analyze a single stock for RSI Mean Reversion signals."""
    if len(df) < 50: return None
    df = df.copy()
    
    # Calculate Indicators
    df['RSI_2'] = calculate_rsi(df['Close'], period=2)
    df['trend_sma'] = df['Close'].rolling(window=200).mean()
    
    signal_type = detect_rsi_signal(df)
    
    if signal_type:
        current_price = df['Close'].iloc[-1]
        if not pd.isna(df['trend_sma'].iloc[-1]) and current_price < df['trend_sma'].iloc[-1]:
             return None 
             
        sma_value = df['trend_sma'].iloc[-1] if not pd.isna(df['trend_sma'].iloc[-1]) else current_price
        volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].tail(20).mean()
        daily_change = ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100
        rsi_val = df['RSI_2'].iloc[-1]
        pct_from_sma = ((current_price - sma_value) / sma_value) * 100

        return Signal(
            symbol=ticker,
            signal_type=signal_type,
            price=float(current_price),
            sma=float(rsi_val), 
            pct_from_sma=float(pct_from_sma),
            volume_ratio=float(volume / avg_volume) if avg_volume > 0 else 0.0,
            daily_change_pct=float(daily_change),
            timestamp=datetime.now()
        )
    return None

# =============================================================================
# DATA FETCHING
# =============================================================================

async def fetch_data_yahoo(tickers: List[str]) -> dict:
    """Fetch data from Yahoo Finance."""
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            if not df.empty and len(df) >= 200:
                data[ticker] = df
        except Exception as e:
            print(f"   ❌ Error fetching {ticker} from Yahoo: {e}")
    return data

async def fetch_data_alpaca(tickers: List[str], api_key: str, secret_key: str) -> dict:
    """Fetch data from Alpaca."""
    client = StockHistoricalDataClient(api_key, secret_key)
    tickers_alpaca = [t.replace('-', '.') for t in tickers]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    data = {}
    chunk_size = 50
    for i in range(0, len(tickers_alpaca), chunk_size):
        chunk = tickers_alpaca[i:i + chunk_size]
        try:
            request = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day, start=start_date, end=end_date, feed=DataFeed.IEX)
            bars = client.get_stock_bars(request)
            for symbol in chunk:
                if symbol in bars.data:
                    df = pd.DataFrame([{'Open': b.open, 'High': b.high, 'Low': b.low, 'Close': b.close, 'Volume': b.volume} for b in bars.data[symbol]])
                    if not df.empty:
                        # Map back to the original ticker with dash if needed
                        original_ticker = symbol.replace('.', '-')
                        data[original_ticker] = df
        except Exception:
            continue
    return data

async def scan_for_signals(config: TradingConfig = DEFAULT_CONFIG) -> List[Signal]:
    """Scan S&P 500 using config-driven parameters."""
    import os
    tickers = get_sp500_tickers()
    tickers = [t for t in tickers if t not in config.blacklist_tickers]
    
    # Fetch data
    api_key = config.alpaca_api_key
    secret_key = config.alpaca_secret_key
    
    if ALPACA_DATA_AVAILABLE and api_key and secret_key:
        data = await fetch_data_alpaca(tickers, api_key, secret_key)
    elif YFINANCE_AVAILABLE:
        data = await fetch_data_yahoo(tickers)
    else:
        raise RuntimeError("No data source available")
    
    signals = []
    print(f"   🔎 Analyzing Market: Checking {len(data)} stocks for opportunities...")
    
    for ticker, df in data.items():
        signal = analyze_stock(ticker, df, config)
        if signal:
            # Apply filters from Config
            if signal.volume_ratio < config.min_volume_ratio: continue
            if signal.price < config.min_stock_price or signal.price > config.max_stock_price: continue
            
            signals.append(signal)
            print(f"   ✨ Potential Opportunity: {ticker} ({signal.signal_type})")
    
    signals.sort(key=lambda s: s.volume_ratio, reverse=True)
    return signals

