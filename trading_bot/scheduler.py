"""
Trading Scheduler
==================
Handles timing and scheduling of trading operations.
"""

import asyncio
from datetime import datetime, time
from typing import Optional
import pytz

from langgraph.graph import StateGraph

from trading_bot.config import TradingConfig
from trading_bot.graph import TradingState


class TradingScheduler:
    """
    Manages the trading schedule and runs the bot during market hours.
    
    Schedule:
    - Pre-market scan: 9:25 AM ET
    - Market open: 9:30 AM ET
    - Continuous monitoring: Every scan_interval during market hours
    - Close positions: 3:55 PM ET
    - Market close: 4:00 PM ET
    """
    
    def __init__(self, graph: StateGraph, config: TradingConfig):
        self.graph = graph
        self.config = config
        self.et_tz = pytz.timezone('US/Eastern')
        self.running = False
        self._shutdown_event = asyncio.Event()
    
    def _get_et_time(self) -> datetime:
        """Get current time in Eastern timezone."""
        return self.config.get_now_et()
    
    def _parse_time(self, time_str: str) -> time:
        """Parse time string (HH:MM) to time object."""
        parts = time_str.split(":")
        return time(int(parts[0]), int(parts[1]))
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = self._get_et_time()
        
        # Check if weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        market_open = self._parse_time(self.config.market_open)
        market_close = self._parse_time(self.config.market_close)
        current_time = now.time()
        
        return market_open <= current_time <= market_close
    
    def should_close_positions(self) -> bool:
        """Check if we should close all positions."""
        now = self._get_et_time()
        close_time = self._parse_time(self.config.close_positions_time)
        return now.time() >= close_time
    
    def time_until_market_open(self) -> Optional[float]:
        """Calculate seconds until market opens."""
        now = self._get_et_time()
        
        # If weekend, calculate to Monday
        days_ahead = 0
        if now.weekday() == 5:  # Saturday
            days_ahead = 2
        elif now.weekday() == 6:  # Sunday
            days_ahead = 1
        
        market_open = self._parse_time(self.config.market_open)
        target = now.replace(
            hour=market_open.hour,
            minute=market_open.minute,
            second=0,
            microsecond=0
        )
        
        if days_ahead > 0:
            target = target + timedelta(days=days_ahead)
        elif now.time() > market_open:
            # Market already opened today, wait for tomorrow
            target = target + timedelta(days=1)
            if target.weekday() == 5:  # Skip to Monday
                target = target + timedelta(days=2)
        
        return (target - now).total_seconds()
    
    async def run(self):
        """Main run loop for the trading bot."""
        self.running = True
        
        print("\n🤖 Bot Active. Systems initialized.")
        print(f"   🕒 Server Time (ET): {self._get_et_time().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Settings: Scanning every {self.config.scan_interval_minutes}m | Checking positions every {self.config.position_check_interval_seconds}s")
        
        while self.running:
            try:
                if self.is_market_open():
                    await self._run_trading_loop()
                else:
                    await self._wait_for_market()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"\n   ❌ System Alert: Main loop encountered an issue: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _run_trading_loop(self):
        """Run the trading loop during market hours."""
        print(f"\n📈 Market is OPEN. Trading session active.")
        
        # Capture starting equity for the day
        start_equity = 0.0
        try:
            from alpaca.trading.client import TradingClient
            client = TradingClient(self.config.alpaca_api_key, self.config.alpaca_secret_key, paper=self.config.paper_trading)
            acct = client.get_account()
            start_equity = float(acct.equity)
            print(f"   💰 Starting Equity: ${start_equity:,.2f}")
        except Exception as e:
            print(f"   ⚠️ Could not fetch starting equity: {e}")
        
        # Initial state
        state: TradingState = {
            "messages": [],
            "signals": [],
            "positions": [],
            "buying_power": 0,
            "portfolio_value": 0,
            "daily_pnl": 0,
            "is_market_open": True,
            "should_close_all": False,
            "daily_loss_limit_hit": False,
            "perform_scan": True,  # Always scan on startup
            "current_action": None,
            "last_error": None,
        }
        
        last_scan_time = None
        
        while self.is_market_open() and self.running:
            # Generate a new thread_id for each cycle to ensure clean conversation history
            # This prevents the context window (and token usage) from growing infinitely
            thread_id = f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": self.config.recursion_limit
            }

            # Reset message history for this new cycle
            # We only need current positions/signals (in state), not past chat logs
            state["messages"] = []

            now = self._get_et_time()
            
            # Check if we should close all positions
            state["should_close_all"] = self.should_close_positions()
            
            # Determine if we should scan
            should_scan = (
                last_scan_time is None or
                (now - last_scan_time).total_seconds() >= self.config.scan_interval_minutes * 60
            )
            
            state["perform_scan"] = should_scan
            
            if should_scan:
                print(f"\n⏰ {now.strftime('%H:%M:%S')} - Running full market scan...")
                last_scan_time = now
            else:
                # Verbose logging (commented out to avoid spam, or print simple dot)
                # print(f"   {now.strftime('%H:%M:%S')} - Checking positions...", end="\r")
                pass
                
            try:
                # Invoke the graph
                result = await self.graph.ainvoke(state, config)
                
                # Update state with results
                state.update(result)
                
                # Log status if important change or periodic Scan
                if should_scan or state.get("current_action") != "monitoring_positions":
                     self._log_status(state)
                
            except Exception as e:
                print(f"   ⚠️  Cycle Alert: Minor issue in trading cycle: {e}")
            
            # Short sleep for position checking
            await asyncio.sleep(self.config.position_check_interval_seconds)
        
        print(f"\n📉 Market Closed")
        await self._generate_daily_report(start_equity)
    
    async def _generate_daily_report(self, start_equity: float):
        """Generate and save the daily trading report/journal AND structured memory."""
        print("   📝 Generating Daily Journal & Updating Memory...")
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            from trading_bot.memory import TradingMemory, DailyEpisode
            import os

            # Initialize Memory with persistent path
            memory = TradingMemory(data_dir=self.config.data_dir)

            client = TradingClient(self.config.alpaca_api_key, self.config.alpaca_secret_key, paper=self.config.paper_trading)
            acct = client.get_account()
            end_equity = float(acct.equity)
            
            # Get filled orders from today
            today = self._get_et_time().date()
            orders = client.get_orders(filter=GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=100,
                after=datetime.combine(today, time.min)
            ))
            today_orders = [o for o in orders if o.filled_at and o.filled_at.date() == today]
            
            # Identify the "Champion" stock (the one we traded locally)
            traded_symbols = set(o.symbol for o in today_orders)
            champion = ", ".join(traded_symbols) if traded_symbols else "NONE (No Trades Taken)"
            
            pnl = end_equity - start_equity
            pnl_pct = (pnl / start_equity) * 100 if start_equity > 0 else 0
            
            # --- 1. WRITE TO TEXT JOURNAL (HUMAN READABLE) ---
            emoji = "🟢" if pnl >= 0 else "🔴"
            
            report = f"""
================================================================================
DATE: {today.strftime('%Y-%m-%d')} | {emoji} RESULT: ${pnl:+.2f} ({pnl_pct:+.2f}%)
--------------------------------------------------------------------------------
START EQUITY:   ${start_equity:,.2f}
END EQUITY:     ${end_equity:,.2f}

CHAMPION STOCK: {champion}

TRADES EXECUTED:"""
            
            if not today_orders:
                report += "\n(No trades executed today)"
            else:
                for o in reversed(today_orders): # Oldest first
                    side = o.side.upper()
                    price = float(o.filled_avg_price) if o.filled_avg_price else 0
                    qty = o.qty
                    report += f"\n- {o.filled_at.strftime('%H:%M')} {side} {o.symbol}: {qty} shares @ ${price:.2f}"

            report += "\\n\\nNOTES:\\n(Auto-generated)\\n================================================================================\\n\\n"
            
            # Append to file in persistent directory
            journal_path = os.path.join(self.config.data_dir, "daily_journal.txt")
            with open(journal_path, "a") as f:
                f.write(report)
                
            print(f"   ✅ Journal saved to {journal_path}")

            # --- 2. SAVE TO STRUCTURED MEMORY (AI READABLE) ---
            # Capture the config used today (simplified)
            config_snapshot = {
                "max_position_size": self.config.max_position_size_pct,
                "stop_loss": self.config.stop_loss_pct,
                "take_profit": self.config.take_profit_pct,
                "strategy": "podium_strategy" 
            }

            episode = DailyEpisode(
                date=today.strftime('%Y-%m-%d'),
                config_used=config_snapshot,
                champion_stock=champion,
                start_equity=start_equity,
                end_equity=end_equity,
                pnl=pnl,
                pnl_pct=pnl_pct,
                win=(pnl > 0)
            )
            
            memory.record_episode(episode)
            
        except Exception as e:
            print(f"   ❌ Failed to save daily report/memory: {e}")

    async def _wait_for_market(self):
        """Wait for market to open."""
        wait_seconds = self.time_until_market_open()
        
        if wait_seconds and wait_seconds > 0:
            hours = int(wait_seconds // 3600)
            minutes = int((wait_seconds % 3600) // 60)
            
            print(f"\n💤 Market is closed. Standing by for {hours}h {minutes}m.")
            
            # Sleep in chunks so we can respond to shutdown
            while wait_seconds > 0 and self.running:
                sleep_time = min(wait_seconds, 300)  # Max 5 minute chunks
                await asyncio.sleep(sleep_time)
                wait_seconds -= sleep_time
    
    def _log_status(self, state: TradingState):
        """Log current trading status."""
        positions = state.get("positions", [])
        signals = state.get("signals", [])
        action = state.get("current_action", "unknown")
        
        # Map technical action names to specific descriptions
        action_map = {
            "monitoring_positions": "Monitoring active positions",
            "scanned_for_signals": "Market scan complete",
            "synced_account": "Account data synced",
            "agent_decided": "Analyzing next moves",
            "scan_failed": "Scan encountered an issue",
        }
        
        friendly_action = action_map.get(action, action.replace("_", " ").title())
        
        print(f"   ℹ️  Status: {friendly_action}")
        print(f"   📊 Detail: {len(positions)} positions open. {len(signals)} new signals found.")
        
        if state.get("daily_loss_limit_hit"):
            print(f"   🛑 RISK SAFETY: Daily loss limit reached. Trading paused.")
        
        if state.get("should_close_all"):
            print(f"   ⏰ END OF DAY: Closing all positions now.")
    
    async def shutdown(self):
        """Gracefully shutdown the scheduler."""
        print("\n🛑 Shutting down scheduler...")
        self.running = False
        self._shutdown_event.set()


# Need this import for timedelta
from datetime import timedelta
