from trading_bot.config import DEFAULT_CONFIG

# ... (rest of imports)

class QuantLab:
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config
        self.cache = {} 

    # ... (get_history stays same)

    def run_backtest(self, symbol: str, rsi_threshold: int, stop_loss_pct: float, take_profit_pct: float, days: int = 180) -> Dict:
        """
        Fast backtest of RSI Mean Reversion strategy.
        Strategy: Buy if RSI < Threshold. Sell if Target/Stop hit OR RSI > 70.
        """
        # Ensure we don't exceed the global safety limits in backtesting
        # Stop loss should be AT MOST the config limit
        safe_stop = min(stop_loss_pct, self.config.stop_loss_pct)
        
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
                stop_price = entry_price * (1 - safe_stop)
                target_price = entry_price * (1 + take_profit_pct)
                
                # Check internal bar extremes
                if curr['low'] <= stop_price:
                    trades.append((stop_price - entry_price) / entry_price) # Stopped out
                    in_position = False
                elif curr['high'] >= target_price:
                    trades.append((target_price - entry_price) / entry_price) # Target hit
                    in_position = False
                elif curr['rsi'] > 70:
                    trades.append((curr['close'] - entry_price) / entry_price)
                    in_position = False
            
            elif signal_buy and not in_position:
                in_position = True
                entry_price = curr['close']
                
        # Calculate stats
        if not trades:
            return {"trades": 0, "win_rate": 0.0, "total_return_pct": 0.0, "rating": "NEUTRAL"}
            
        wins = [t for t in trades if t > 0]
        win_rate = len(wins) / len(trades) if trades else 0
        total_ret = sum(trades)
        
        # Rating
        rating = "POOR"
        if total_ret > 0: rating = "GOOD" 
        if win_rate > 0.6 and total_ret > 0.10: rating = "EXCELLENT"
        if total_ret < -0.05: rating = "DANGEROUS" 
            
        return {
            "trades": len(trades),
            "win_rate": round(win_rate * 100, 1),
            "total_return_pct": round(total_ret * 100, 2),
            "rating": rating
        }

    def optimize_params(self, symbol: str) -> Dict:
        """Runs a mini-grid search to find the best RSI parameters."""
        rsi_thresholds = [5, 10, 15, 20]
        # Only test stops that are equal or TIGHTER than config limit
        stop_losses = [0.01, 0.015, 0.02, self.config.stop_loss_pct]
        stop_losses = sorted(list(set(stop_losses))) # Unique & sorted
        
        best_score = -999
        best_params = {}
        best_metrics = {}
            
        for rsi in rsi_thresholds:
            for sl in stop_losses:
                tp = sl * 2.0 
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

    def get_volatility(self, symbol: str, days: int = 14) -> Dict:
        """Calculate Average True Range (ATR) and Volatility."""
        df = self.get_history(symbol, days=60)
        if df.empty or len(df) < days + 1:
            return {"error": "Insufficient data"}
            
        high = df['high']; low = df['low']; close = df['close'].shift(1)
        tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=days).mean().iloc[-1]
        volatility_std = df['close'].pct_change().rolling(window=days).std().iloc[-1]
        current_price = df['close'].iloc[-1]
        atr_stop_dist = atr * 2.0
        atr_stop_pct = (atr_stop_dist / current_price)
        
        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "atr": round(atr, 2),
            "volatility_std": round(volatility_std, 4),
            "suggested_stop_distance": round(atr_stop_dist, 2),
            "suggested_stop_pct": round(atr_stop_pct, 4),
            "max_allowed_stop_pct": round(self.config.stop_loss_pct, 4), # NEW: context!
            "market_condition": "VOLATILE" if atr_stop_pct > self.config.stop_loss_pct else "STABLE"
        }

# Global Instance
quant_lab = QuantLab(DEFAULT_CONFIG)

def vet_trade_signal(symbol: str, proposed_sma: int = 15) -> str:
    """Wrapper for agent to verify a trade."""
    # Use config-based risk levels for vetting
    res = quant_lab.run_backtest(symbol, 15, DEFAULT_CONFIG.stop_loss_pct, DEFAULT_CONFIG.take_profit_pct) 
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
        return json.dumps({
            "message": f"OPTIMIZATION RESULT: Use RSI<{p['rsi_threshold']}, Stop Loss {p['stop_pct']:.2%}, Target {p['take_profit_pct']:.2%}.",
            "data": res
        })
    return "Optimization failed: No profitable strategy found for this stock."

def get_volatility_metrics(symbol: str) -> str:
    """Wrapper for agent to get volatility data."""
    res = quant_lab.get_volatility(symbol)
    if "error" in res: return f"Could not calculate volatility for {symbol}."
    return json.dumps(res, indent=2)
