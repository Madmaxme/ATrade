"""
Quant Lab
=========
Implements "Just-in-Time Backtesting" and "Dynamic Parameter Optimization".
This module allows the agent to vet potential trades against historical data
to ensure the strategy is robust for the specific asset.

Hybridization of:
- Friend's "Evolutionary Optimization" (Mini-Grid Search)
- Your "Contextual Intelligence" (LLM Decision Making)
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Setup logger
logger = logging.getLogger("QuantLab")

class QuantLab:
    def __init__(self):
        self.cache = {} # Simple in-memory cache for dataframes

    def get_history(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Fetch historical data for a symbol.
        Uses yfinance for broad coverage. 
        """
        # Clean symbol (e.g. BRK-B -> BRK.B for Yahoo vs Alpaca differences)
        # Yahoo uses '-' (BRK-B), Alpaca uses '.' (BRK.B) if we were using Alpaca.
        # But wait, user said earlier Alpaca uses dots...
        # Actually yfinance uses BRK-B.
        
        yf_symbol = symbol.replace('.', '-') # Ensure Yahoo format
        
        if yf_symbol in self.cache:
            # Check if cache is fresh (simple check: if it has today's date?)
            # For this prototype, we just return cached if it exists for this run
            return self.cache[yf_symbol]
            
        try:
            # Download with buffer
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 50) # +50 for SMA warmup
            
            df = yf.download(yf_symbol, start=start_date, end=end_date, progress=False, interval="1d")
            
            if df.empty:
                return pd.DataFrame()
            
            # Normalize columns (Yahoo sometimes returns MultiIndex if multiple tickers)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            # Rename to standard internal format
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
            })
            
            # Fill missing
            df = df.interpolate(method='linear')
            
            self.cache[yf_symbol] = df
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch history for {symbol}: {e}")
            return pd.DataFrame()

    def run_backtest(self, symbol: str, rsi_threshold: int, stop_loss_pct: float, take_profit_pct: float, days: int = 180) -> Dict:
        """
        Fast backtest of RSI Mean Reversion strategy.
        Strategy: Buy if RSI < Threshold. Sell if Target/Stop hit OR RSI > 70.
        """
        df = self.get_history(symbol, days=days)
        if df.empty or len(df) < 50:
            return {"error": "Insufficient data"}
            
        # 1. Calculate Indicators
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        trades = []
        in_position = False
        entry_price = 0.0
        
        records = df.iloc[2:].to_dict('records') # Skip first 2 NaN RSI
        
        for i in range(1, len(records)):
            curr = records[i]
            prev = records[i-1]
            
            # RSI Buy Logic: RSI drops below threshold (Oversold)
            signal_buy = prev['rsi'] < rsi_threshold
            
            if in_position:
                # Check stops
                stop_price = entry_price * (1 - stop_loss_pct)
                target_price = entry_price * (1 + take_profit_pct)
                
                # Check internal bar extremes
                if curr['low'] <= stop_price:
                    trades.append((stop_price - entry_price) / entry_price) # Stopped out
                    in_position = False
                elif curr['high'] >= target_price:
                    trades.append((target_price - entry_price) / entry_price) # Target hit
                    in_position = False
                elif curr['rsi'] > 70:
                    # Logic: Sell the "Rip" (RSI Reversion complete)
                    trades.append((curr['close'] - entry_price) / entry_price)
                    in_position = False
                else:
                    pass # Hold
            
            elif signal_buy and not in_position:
                in_position = True
                entry_price = curr['close']
                
        # Calculate stats
        if not trades:
            return {
                "trades": 0,
                "win_rate": 0.0,
                "total_return_pct": 0.0,
                "rating": "NEUTRAL"
            }
            
        wins = [t for t in trades if t > 0]
        
        win_rate = len(wins) / len(trades) if trades else 0
        total_ret = sum(trades)
        
        # Rating
        rating = "POOR"
        if total_ret > 0: rating = "GOOD" # Profitable
        if win_rate > 0.6 and total_ret > 0.10: rating = "EXCELLENT" # High Alpha
        if total_ret < -0.05: rating = "DANGEROUS" # Heavy drawdown
            
        return {
            "trades": len(trades),
            "win_rate": round(win_rate * 100, 1),
            "total_return_pct": round(total_ret * 100, 2),
            "rating": rating
        }

    def optimize_params(self, symbol: str) -> Dict:
        """
        Runs a mini-grid search to find the best RSI parameters for THIS symbol.
        """
        # Parameter Space
        rsi_thresholds = [5, 10, 15, 20, 25] # Deep dip vs Shallow dip
        stop_losses = [0.02, 0.03, 0.05, 0.07]
        
        best_score = -999
        best_params = {}
        best_metrics = {}
            
        # Run Grid
        for rsi in rsi_thresholds:
            for sl in stop_losses:
                tp = sl * 2.0 
                # Backtest
                res = self.run_backtest(symbol, rsi, sl, tp, days=180)
                score = res.get('total_return_pct', -100)
                
                if score > best_score:
                    best_score = score
                    best_params = {"rsi_threshold": rsi, "stop_pct": sl, "take_profit_pct": tp}
                    best_metrics = res
        
        if not best_params or best_metrics['total_return_pct'] <= 0:
            return {"status": "FAILED", "reason": "No profitable strategy found"}
            
        return {
            "status": "OPTIMIZED",
            "symbol": symbol,
            "recommended_params": best_params,
            "metrics": best_metrics,
            "message": f"Best fit for {symbol}: Buy RSI<{best_params['rsi_threshold']}, Stop {best_params['stop_pct']:.2%}. (Return: {best_metrics['total_return_pct']}%)"
        }

# Global Instance
quant_lab = QuantLab()

def vet_trade_signal(symbol: str, proposed_sma: int = 15) -> str:
    """Wrapper for agent to verify a trade. (proposed_sma argument ignored, treated as threshold default)"""
    # Default to RSI < 15 check
    res = quant_lab.run_backtest(symbol, 15, 0.03, 0.06) 
    
    if res.get('rating') in ["DANGEROUS", "POOR"]:
         return f"VETO: Historical backtest shows this strategy loses money on {symbol} (Return: {res['total_return_pct']}%)."
    elif res.get('trades', 0) == 0:
         return f"WARNING: No historical signals found for {symbol} to verify strategy."
    else:
         return f"APPROVED: Backtest indicates robustness (Win Rate: {res['win_rate']}%, Return: {res['total_return_pct']}%)."

def find_best_settings(symbol: str) -> str:
    """Wrapper for agent to ask for optimization."""
    res = quant_lab.optimize_params(symbol)
    if res.get('status') == 'OPTIMIZED':
        p = res['recommended_params']
        # Return a format that is readable by BOTH LLM and Regex/JSON parser
        return json.dumps({
            "message": f"OPTIMIZATION RESULT: Use RSI<{p['rsi_threshold']}, Stop Loss {p['stop_pct']:.2%}, Target {p['take_profit_pct']:.2%}.",
            "data": res
        })
    else:
        return "Optimization failed: No profitable strategy found for this stock."
