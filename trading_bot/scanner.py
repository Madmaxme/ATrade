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

def calculate_sma(prices: pd.Series, window: int = 21) -> pd.Series:
    """Calculate Simple Moving Average."""
    return prices.rolling(window=window).mean()


def detect_crossover(df: pd.DataFrame, sma_col: str = 'SMA_21') -> Optional[str]:
    """Detect SMA crossover."""
    if len(df) < 2:
        return None
    
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    
    if pd.isna(today[sma_col]) or pd.isna(yesterday[sma_col]):
        return None
    
    # BUY: crossed from below to above
    if yesterday['Close'] <= yesterday[sma_col] and today['Close'] > today[sma_col]:
        return 'BUY'
    
    # SELL: crossed from above to below
    if yesterday['Close'] >= yesterday[sma_col] and today['Close'] < today[sma_col]:
        return 'SELL'
    
    return None


def analyze_stock(ticker: str, df: pd.DataFrame, sma_period: int = 21) -> Optional[Signal]:
    """Analyze a single stock for crossover signals."""
    if len(df) < sma_period + 1:
        return None
    
    df = df.copy()
    df['SMA_21'] = calculate_sma(df['Close'], window=sma_period)
    
    signal_type = detect_crossover(df)
    
    if signal_type:
        current_price = df['Close'].iloc[-1]
        sma_value = df['SMA_21'].iloc[-1]
        volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].tail(sma_period).mean()
        daily_change = ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100
        pct_from_sma = ((current_price - sma_value) / sma_value) * 100
        
        return Signal(
            symbol=ticker,
            signal_type=signal_type,
            price=float(current_price),
            sma=float(sma_value),
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
            df = stock.history(period="2mo")
            if not df.empty and len(df) >= 22:
                data[ticker] = df
        except Exception as e:
            print(f"   ❌ Error fetching {ticker} from Yahoo: {e}")
            continue
    
    print(f"   ✓ Market data download complete ({len(data)} stocks)")
    
    return data


async def fetch_data_alpaca(tickers: List[str], api_key: str, secret_key: str) -> dict:
    """Fetch data from Alpaca."""
    client = StockHistoricalDataClient(api_key, secret_key)
    
    # Alpaca expects dots for classes (e.g., BRK.B), not dashes
    tickers = [t.replace('-', '.') for t in tickers]
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=45)
    
    data = {}
    chunk_size = 50
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        print(f"   Downloading market data: Part {i//chunk_size + 1}/{(len(tickers)-1)//chunk_size + 1} ({len(chunk)} stocks)...")
        
        try:
            request = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
                feed=DataFeed.IEX
            )
            
            bars = client.get_stock_bars(request)
            
            for symbol in chunk:
                if symbol in bars.data:
                    symbol_bars = bars.data[symbol]
                    df = pd.DataFrame([{
                        'Open': bar.open,
                        'High': bar.high,
                        'Low': bar.low,
                        'Close': bar.close,
                        'Volume': bar.volume,
                    } for bar in symbol_bars])
                    
                    if not df.empty:
                        data[symbol] = df
            
        except Exception as e:
            print(f"   ❌ Network Issue: Could not fetch signals for chunk {chunk[0]}...: {e}")
            continue
    
    print(f"   ✓ Market Data Ready: loaded {len(data)} stocks")
    return data


# =============================================================================
# MAIN SCAN FUNCTION
# =============================================================================

async def scan_for_signals(
    sma_period: int = 21,
    min_volume_ratio: float = 0.3,
    min_price: float = 10.0,
    max_price: float = 500.0,
    blacklist: List[str] = None
) -> List[Signal]:
    """
    Scan S&P 500 for SMA crossover signals.
    
    Returns list of Signal objects sorted by volume ratio.
    """
    import os
    
    blacklist = blacklist or []
    
    # Get tickers
    tickers = get_sp500_tickers()
    tickers = [t for t in tickers if t not in blacklist]
    
    # Fetch data
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if ALPACA_DATA_AVAILABLE and api_key and secret_key:
        data = await fetch_data_alpaca(tickers, api_key, secret_key)
    elif YFINANCE_AVAILABLE:
        data = await fetch_data_yahoo(tickers)
    else:
        raise RuntimeError("No data source available")
    
    # Scan for signals
    signals = []
    
    print(f"   🔎 Analyzing Market: Checking {len(data)} stocks for opportunities...")
    
    for ticker, df in data.items():
        signal = analyze_stock(ticker, df, sma_period)
        
        if signal:
            # Apply filters
            if signal.volume_ratio < min_volume_ratio:
                # print(f"     Skipping {ticker}: Low volume ratio ({signal.volume_ratio:.2f})")
                continue
            if signal.price < min_price or signal.price > max_price:
                # print(f"     Skipping {ticker}: Price out of range (${signal.price:.2f})")
                continue
            
            signals.append(signal)
            print(f"   ✨ Potential Opportunity: {ticker} ({signal.signal_type})")
    
    print(f"   ✅ Scan Complete: Identified {len(signals)} actionable signals.")

    # Sort by volume ratio (highest conviction first)
    signals.sort(key=lambda s: s.volume_ratio, reverse=True)
    
    return signals
