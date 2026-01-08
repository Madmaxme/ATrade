"""
Agent Simulator (Mode B)
========================
This module runs the full LangGraph agent against historical data.
It mocks the environment (Alpaca, Time, News) so the agent "experiences" past market conditions.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dataclasses import dataclass, field

from langchain_core.tools import tool
from langgraph.graph import StateGraph

from trading_bot.config import TradingConfig
from trading_bot.graph import create_trading_graph, TradingState
from trading_bot.backtester import Backtester  # Reuse data fetching

@dataclass
class SimulationContext:
    """Holds the state of the simulated world."""
    current_time: datetime
    equity: float
    cash: float
    positions: Dict[str, dict]  # Symbol -> Position details
    market_data: Dict[str, pd.DataFrame]  # Symbol -> OHLCV history
    orders: List[dict] = field(default_factory=list)
    
    def get_current_price(self, symbol: str) -> float:
        """Get the price of a stock at the current simulated time."""
        if symbol not in self.market_data:
            return 0.0
            
        df = self.market_data[symbol]
        # Find the row for the current date
        # Assuming Daily bars for now, so we return the "Open" or "Close" depending on time
        # For simplicity in daily simulation, we return 'Close' of the day (hindsight) 
        # OR 'Open' if it's morning.
        
        # Proper way: Filter for date <= current_time.date()
        mask = df.index.date <= self.current_time.date()
        history = df[mask]
        
        if history.empty:
            return 0.0
            
        # Get the latest bar
        bar = history.iloc[-1]
        
        # If simulation timestamp matches the bar date, user might want Open/Close
        # depending on intra-day timing.
        # Let's assume we are trading at "Close" prices for simplicity of this prototype
        return float(bar['Close'])

class AgentSimulator:
    def __init__(self, config: TradingConfig, initial_capital: float = 100000.0):
        self.config = config
        self.context = SimulationContext(
            current_time=datetime.now(), # Will be set during run
            equity=initial_capital,
            cash=initial_capital,
            positions={},
            market_data={}
        )
        self.graph = None
        
    async def setup(self, tickers: List[str], days_back: int = 30):
        """Prepare data and the agent graph."""
        # 1. Fetch Data (reusing backtester logic)
        # We start a temporary backtester to just download data
        bt = Backtester(self.config)
        await bt.fetch_data(tickers, days_back)
        self.context.market_data = bt.market_data
        
        # 2. Create Mock Tools
        mock_tools = self._create_mock_tools()
        
        # 3. Build Graph with Mock Tools
        self.graph = await create_trading_graph(self.config, override_tools=mock_tools)
        
    def _create_mock_tools(self) -> List:
        """Create tools that interact with SimulationContext instead of APIs."""
        
        # We need to define these functions inside here to access 'self.context' 
        # or bind them to self.
        
        @tool
        def mcp_alpaca_get_account_info():
            """Retrieves account information."""
            return f"""
            Account ID: SIMULATED_ACCOUNT
            Status: ACTIVE
            Buying Power: ${self.context.cash:,.2f}
            Cash: ${self.context.cash:,.2f}
            Equity: ${self.context.equity:,.2f}
            """
            
        @tool
        def mcp_alpaca_get_all_positions():
            """Retrieves all current positions."""
            pos_list = []
            for sym, p in self.context.positions.items():
                curr_price = self.context.get_current_price(sym)
                market_val = p['qty'] * curr_price
                pl = market_val - (p['qty'] * p['entry_price'])
                
                pos_list.append(
                    f"Symbol: {sym}, Qty: {p['qty']}, Side: long, "
                    f"Entry: ${p['entry_price']:.2f}, Current: ${curr_price:.2f}, "
                    f"Market Val: ${market_val:.2f}, P/L: ${pl:.2f}"
                )
            return "\n".join(pos_list) if pos_list else "No open positions."
            
        @tool
        def mcp_alpaca_place_stock_order(symbol: str, side: str, quantity: float, type: str = "market", time_in_force: str = "day"):
            """Places a stock order (Simulated)."""
            price = self.context.get_current_price(symbol)
            if price == 0:
                return f"Error: No price data for {symbol}"
                
            cost = price * quantity
            
            if side == 'buy':
                if self.context.cash < cost:
                    return "Error: Insufficient buying power"
                
                self.context.cash -= cost
                if symbol in self.context.positions:
                    # Average down logic omitted for brevity
                    self.context.positions[symbol]['qty'] += quantity
                else:
                    self.context.positions[symbol] = {'qty': quantity, 'entry_price': price}
                    
                return f"Filled: BUY {quantity} {symbol} @ ${price:.2f}"
                
            elif side == 'sell':
                curr = self.context.positions.get(symbol)
                if not curr or curr['qty'] < quantity:
                    return "Error: Not enough shares"
                
                revenue = price * quantity
                self.context.cash += revenue
                
                curr['qty'] -= quantity
                if curr['qty'] <= 0:
                    del self.context.positions[symbol]
                    
                return f"Filled: SELL {quantity} {symbol} @ ${price:.2f}"
            
            return "Order received"

        # Import local tools that are pure logic (calculators)
        from trading_bot.tools import (
            calculate_position_size, 
            calculate_stop_and_target, 
            check_daily_loss_limit,
            format_trade_log
        )
        
        # We also need signal evaluation, but we might want to mock "Tavily" sentiment
        @tool
        def get_market_sentiment(symbol: str) -> dict:
            """Get market sentiment (Mocked for Simulation)."""
            # In a real sim, we could load historical headlines.
            # Here we just return neutral/random or skipped.
            return {
                "symbol": symbol,
                "headline_count": 0,
                "top_news": ["(Simulation) Historical news not available in this mode"],
                "sentiment": "NEUTRAL"
            }

        return [
            mcp_alpaca_get_account_info,
            mcp_alpaca_get_all_positions,
            mcp_alpaca_place_stock_order,
            calculate_position_size,
            calculate_stop_and_target,
            check_daily_loss_limit,
            format_trade_log,
            get_market_sentiment
        ]

    async def run(self):
        """Run the simulation loop."""
        if not self.context.market_data:
            print("❌ No data loaded. Run setup() first.")
            return

        sorted_dates = sorted(list(set().union(*[df.index.date for df in self.context.market_data.values()])))
        print(f"🤖 Agent Simulation Starting: {len(sorted_dates)} trading days")
        
        # Initial State
        state = {
            "messages": [],
            "signals": [],
            "positions": [],
            "buying_power": self.context.cash,
            "portfolio_value": self.context.equity,
            "daily_pnl": 0,
            "is_market_open": True,
            "should_close_all": False,
            "daily_loss_limit_hit": False,
            "perform_scan": True,
            "current_action": None,
            "last_error": None,
        }
        
        config = {"recursion_limit": 50}
        
        for sim_date in sorted_dates:
            # Update Context Time
            # We set time to 10:00 AM on that day
            self.context.current_time = datetime.combine(sim_date, datetime.min.time()).replace(hour=10)
            
            print(f"\n📅 SIMULATION DATE: {sim_date}")
            
            # 1. Update Portfolio Value (Mark to Market)
            current_equity = self.context.cash
            for sym, pos in self.context.positions.items():
                p = self.context.get_current_price(sym)
                current_equity += pos['qty'] * p
            self.context.equity = current_equity
            
            # Setup State for this turn
            state["positions"] = [] # The 'account_sync' node will fill this from our Mock Tool
            state["buying_power"] = self.context.cash
            state["portfolio_value"] = self.context.equity
            state["perform_scan"] = True
            
            # 2. Inject Signals?
            # In the real bot, 'scanner' node calls 'scan_for_signals'.
            # That function fetches data from APIs. We need to mock that too?
            # OR we can just pre-calculate signals like Backtester did and inject them into state['signals']
            # bypassing the scanner node?
            
            # Let's bypass scanner for simplicity and inject signals manually based on our Backtester logic
            # This is "Mode B lite"
            
            daily_signals = self._generate_signals_for_date(sim_date)
            state["signals"] = daily_signals
            state["current_action"] = "scanned_for_signals" # Skip scanner node practically
            
            # 3. Run Agent (Agent Node -> Tools -> Agent...)
            # We start at 'account_sync' to refresh positions from our Mock Context
            # Then 'agent' decides
            
            # Since we can't easily start graph at specific node in compiled stategraph without specific configuration,
            # we just run the graph. 
            # BUT the scanner node inside graph calls 'scan_for_signals' which uses real APIs.
            # We need to mock 'scan_for_signals' inside `trading_bot.graph` or `scanner.py`.
            # This is getting tricky with imports.
            
            # Workaround: We passed tools, but 'scanner' is a python function node, not a tool.
            # We can check `state['perform_scan']`. If we set it to False, scanner node does nothing.
            state["perform_scan"] = False # Disable real scanner
            # So we manually injected signals above.
            
            # Generate a thread ID for storage
            thread_id = f"sim_{sim_date.strftime('%Y%m%d')}"
            run_config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 50
            }
            
            try:
                # Run the graph!
                result = await self.graph.ainvoke(state, run_config)
                state = result # Update state for next day/step? 
                
                # Reset messages for next day to avoid context overflow?
                state["messages"] = [] 
                
            except Exception as e:
                print(f"   ❌ Agent Error: {e}")
                
    def _generate_signals_for_date(self, date) -> List[dict]:
        """Generate signals using backtester logic for the specific date."""
        signals = []
        for sym, df in self.context.market_data.items():
             # Logic from mode A
             mask = df.index.date <= date
             hist = df[mask]
             if len(hist) < 22: continue
             
             # Calculate SMA if not present (Backtester adds it, but let's be safe)
             if 'SMA' not in hist.columns:
                 hist = hist.copy()
                 hist['SMA'] = hist['Close'].rolling(21).mean()
             
             curr = hist.iloc[-1]
             prev = hist.iloc[-2]
             
             if prev['Close'] <= prev['SMA'] and curr['Close'] > curr['SMA']:
                 # Signal!
                 signals.append({
                     "symbol": sym,
                     "signal_type": "BUY",
                     "price": float(curr['Close']),
                     "sma": float(curr['SMA']),
                     "pct_from_sma": 0.5, # approx
                     "volume_ratio": 2.0, # dummy high conviction
                     "daily_change_pct": 1.5,
                     "timestamp": self.context.current_time
                 })
        
        return signals[:3] # Return top 3
