"""
Backtester Module
=================
Implements "Mode A" backtesting: pure strategy simulation using historical data.
This allows for rapid testing of strategy parameters (SMA period, SL/TP ratios)
without using live API calls or LLM inference.
"""

import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from trading_bot.config import TradingConfig, DEFAULT_CONFIG

@dataclass
class BacktestTrade:
    symbol: str
    entry_date: datetime
    entry_price: float
    side: str
    quantity: int
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0

class Backtester:
    def __init__(self, config: TradingConfig = DEFAULT_CONFIG, initial_capital: float = 100000.0):
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades: List[BacktestTrade] = []
        self.market_data = {}  # Store DF per ticker
        
    async def fetch_data(self, tickers: List[str], days_back: int = 180):
        """Fetch historical data for backtesting."""
        api_key = self.config.alpaca_api_key
        secret_key = self.config.alpaca_secret_key
        
        if not api_key or not secret_key:
            raise ValueError("Alpaca credentials not found in environment")
            
        print(f"   ⏳ Backtester: Fetching {days_back} days of history for {len(tickers)} stocks...")
        
        client = StockHistoricalDataClient(api_key, secret_key)
        
        # Alpaca expects dots instead of dashes
        formatted_tickers = [t.replace('-', '.') for t in tickers]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            # Fetch in chunks to avoid timeouts
            chunk_size = 50
            for i in range(0, len(formatted_tickers), chunk_size):
                chunk = formatted_tickers[i:i + chunk_size]
                
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
                        # Convert to DataFrame
                        stock_bars = bars.data[symbol]
                        df = pd.DataFrame([{
                            'timestamp': bar.timestamp,
                            'Open': bar.open,
                            'High': bar.high,
                            'Low': bar.low,
                            'Close': bar.close,
                            'Volume': bar.volume,
                        } for bar in stock_bars])
                        
                        # Set index to timestamp
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        
                        # Calculate Indicators directly here
                        df['SMA'] = df['Close'].rolling(window=self.config.sma_period).mean()
                        
                        # Clean symbol back to original (e.g. BRK.B -> BRK-B if needed)
                        # But we just use what we fetched
                        self.market_data[symbol] = df
                        
            print(f"   ✓ Backtester: Data ready. Loaded {len(self.market_data)} stocks.")
            
        except Exception as e:
            print(f"   ❌ Backtester Error: Failed to fetch data: {e}")

    def run(self):
        """Run the simulation event loop."""
        print(f"   🚀 Starting Simulation (Strategy: SMA-{self.config.sma_period} Crossover)...")
        print(f"      Risk Settings: Stop {self.config.stop_loss_pct*100}% | Target {self.config.take_profit_pct*100}% | Max Pos {self.config.max_positions}")
        
        # Align all data to a common date index
        if not self.market_data:
            print("   ⚠️ No data to backtest.")
            return

        # Get all unique dates from all dataframes
        all_dates = sorted(list(set().union(*[df.index.date for df in self.market_data.values()])))
        
        active_positions: List[BacktestTrade] = []
        
        # Iterate through every day in history
        for current_date in all_dates:
            # 1. Manage Active Positions (Exit Logic)
            # We assume we hold mainly intraday, but for backtesting daily bars, 
            # we check if High/Low hit our TP/SL during this day.
            
            remaining_positions = []
            for trade in active_positions:
                # Get today's bar for this stock
                df = self.market_data.get(trade.symbol)
                # Handle cases where current_date is missing for this stock
                try: 
                     # Need to convert date to timestamp/string lookups properly or use reindexed DFs
                     # Simpler: Filter by date
                     today_bar = df[df.index.date == current_date]
                except:
                     today_bar = pd.DataFrame()
                     
                if today_bar.empty:
                    # No data for today (holiday? halted?), hold position
                    remaining_positions.append(trade)
                    continue
                
                bar = today_bar.iloc[0]
                high = bar['High']
                low = bar['Low']
                close = bar['Close']
                
                # Check stops/targets
                # Assumption: If Low < P_stop, we hit stop. 
                # Determine hit prices
                stop_price = trade.entry_price * (1 - self.config.stop_loss_pct) if trade.side == 'long' else trade.entry_price * (1 + self.config.stop_loss_pct)
                target_price = trade.entry_price * (1 + self.config.take_profit_pct) if trade.side == 'long' else trade.entry_price * (1 - self.config.take_profit_pct)
                
                exit_triggered = False
                exit_price = 0.0
                reason = ""
                
                # Logic: Did we hit SL?
                if (trade.side == 'long' and low <= stop_price) or (trade.side == 'short' and high >= stop_price):
                    exit_triggered = True
                    exit_price = stop_price
                    reason = "STOP_LOSS"
                # Logic: Did we hit TP? (If both hit, we assume SL hit first for conservative testing, unless we have minute data)
                elif (trade.side == 'long' and high >= target_price) or (trade.side == 'short' and low <= target_price):
                    exit_triggered = True
                    exit_price = target_price
                    reason = "TAKE_PROFIT"
                # Logic: End of Day close (if it's a day trading bot)
                # The config says "close_positions_time", so we technically close same day
                # For this daily simulation, we can assume we close at 'Close' if neither SL/TP hit
                else:
                    exit_triggered = True
                    exit_price = close
                    reason = "EOD_CLOSE"
                
                if exit_triggered:
                    trade.exit_date = current_date
                    trade.exit_price = exit_price
                    trade.exit_reason = reason
                    
                    # Calculate PnL
                    if trade.side == 'long':
                        trade.pnl = (exit_price - trade.entry_price) * trade.quantity
                        trade.pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
                    else:
                        trade.pnl = (trade.entry_price - exit_price) * trade.quantity
                        trade.pnl_pct = (trade.entry_price - exit_price) / trade.entry_price
                        
                    self.current_capital += trade.pnl
                    self.trades.append(trade) # Archive completed trade
                else:
                    remaining_positions.append(trade)
            
            active_positions = remaining_positions
            
            # 2. Scan for New Entries (Entry Logic)
            # Only if we have capacity
            if len(active_positions) >= self.config.max_positions:
                continue
                
            candidates = []
            
            for symbol, df in self.market_data.items():
                if symbol in [p.symbol for p in active_positions]:
                    continue
                
                # We need "yesterday" and "today" relative to current_simulated_date
                # Since we are iterating dates, we look at the row for 'current_date'
                # But actually, signals are generated Pre-Market or at Open based on Previous Day's close?
                # The 'scanner.py' uses: yesterday <= SMA and today > SMA. 
                # But 'today' in scanner is current live price.
                # In backtest: 'today' is the day we are simulating. We can use Open price to simulate entry.
                # So we check: Yesterday(Close) vs SMA AND Today(Open) vs SMA
                
                # Get indices up to current date
                mask = df.index.date <= current_date
                history_df = df[mask]
                
                if len(history_df) < 2:
                    continue
                    
                prev_bar = history_df.iloc[-2]
                curr_bar = history_df.iloc[-1] # This is 'today'
                
                # Retrieve pre-calculated SMA
                if pd.isna(prev_bar['SMA']) or pd.isna(curr_bar['SMA']):
                    continue
                    
                # CROSSOVER LOGIC
                # We use Prev Close and Current Open to decide entry at Open
                # BUY: Previous Close < Prev SMA  AND  Current Open > Current SMA (Strong Gap Up?)
                # OR stricter: just use previous day's crossover to enter at Open?
                
                # Standard Crossover: Line crosses line.
                # Let's use the Scanner logic: prev_close <= prev_sma AND curr_close > curr_sma
                # BUT we can't know 'curr_close' at the start of the day.
                # SO: We enter if Yesterday had a crossover.
                # If Yesterday Close > Yesterday SMA AND DayBefore Close < SMA... Then today we Buy at Open.
                
                day_before = history_df.iloc[-3] if len(history_df) >= 3 else None
                if day_before is None: continue
                
                # Check for crossover YESTERDAY
                crossover = False
                if day_before['Close'] <= day_before['SMA'] and prev_bar['Close'] > prev_bar['SMA']:
                    crossover = True
                
                if crossover:
                    # Check Volume filter (using yesterday's volume vs avg)
                    avg_vol = history_df['Volume'].iloc[-22:-1].mean()
                    vol_ratio = prev_bar['Volume'] / avg_vol if avg_vol > 0 else 0
                    
                    if vol_ratio >= self.config.min_volume_ratio:
                        candidates.append({
                            'symbol': symbol,
                            'entry_price': curr_bar['Open'], # Enter at Open
                            'vol_ratio': vol_ratio
                        })

            # Sort by volume conviction (highest ratio first)
            candidates.sort(key=lambda x: x['vol_ratio'], reverse=True)
            
            # Enter trades
            for cand in candidates:
                if len(active_positions) >= self.config.max_positions:
                    break
                
                # Size position
                capital_alloc = self.initial_capital * self.config.max_position_size_pct
                qty = int(capital_alloc / cand['entry_price'])
                
                if qty > 0:
                    new_trade = BacktestTrade(
                        symbol=cand['symbol'],
                        entry_date=current_date,
                        entry_price=cand['entry_price'],
                        side='long', # Scanner only does long crossover for now
                        quantity=qty
                    )
                    active_positions.append(new_trade)
                    # print(f"   ➕ Buying {cand['symbol']} at ${cand['entry_price']:.2f} on {current_date}")

        # End of simulation
        print("\n   🏁 Simulation Complete.")
        self.print_stats()

    def print_stats(self):
        """Print statistics."""
        if not self.trades:
            print("   ⚠️ No trades executed.")
            return

        total_trades = len(self.trades)
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(wins) / total_trades * 100
        
        total_pnl = sum(t.pnl for t in self.trades)
        final_equity = self.current_capital
        return_pct = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        print("\n" + "="*50)
        print("   BACKTEST RESULTS")
        print("="*50)
        print(f"   Period Return:     {return_pct:+.2f}%")
        print(f"   Total P&L:         ${total_pnl:+.2f}")
        print(f"   Final Equity:      ${final_equity:,.2f}")
        print(f"   --------------------------------")
        print(f"   Total Trades:      {total_trades}")
        print(f"   Win Rate:          {win_rate:.1f}%")
        print(f"   Avg Win:           ${np.mean([t.pnl for t in wins]):.2f}" if wins else "   Avg Win: $0")
        print(f"   Avg Loss:          ${np.mean([t.pnl for t in losses]):.2f}" if losses else "   Avg Loss: $0")
        
        # Best trade
        if wins:
            best = max(wins, key=lambda x: x.pnl)
            print(f"   🏆 Best Trade:      {best.symbol} (+${best.pnl:.2f})")
        
        print("\n   Parameters Used:")
        print(f"   SMA: {self.config.sma_period} | Risk: {self.config.stop_loss_pct*100}% | Reward: {self.config.take_profit_pct*100}%")
        print("="*50 + "\n")
