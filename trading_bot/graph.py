"""
Trading Bot Graph Definition
=============================
LangGraph-based state machine for the day trading bot.
"""

import operator
from datetime import datetime
from typing import Annotated, TypedDict, List, Optional, Literal
from dataclasses import dataclass

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from trading_bot.config import TradingConfig
from trading_bot.tools import get_trading_tools
from trading_bot.prompts import TRADER_SYSTEM_PROMPT
from trading_bot.memory import memory_manager


# =============================================================================
# STATE DEFINITIONS
# =============================================================================

@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    order_id: str


@dataclass 
class Signal:
    """Represents a trading signal from the scanner."""
    symbol: str
    signal_type: str  # 'BUY' or 'SELL'
    price: float
    sma: float
    pct_from_sma: float
    volume_ratio: float
    daily_change_pct: float
    timestamp: datetime


class TradingState(TypedDict):
    """State for the trading graph."""
    
    # Message history for the agent
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Current signals from scanner (as dictionaries)
    signals: List[dict]
    
    # Current open positions (as dictionaries)
    positions: List[dict]
    
    # Account info
    buying_power: float
    portfolio_value: float
    daily_pnl: float
    
    # Trading status
    is_market_open: bool
    should_close_all: bool  # True when approaching market close
    daily_loss_limit_hit: bool
    
    # Control flags
    perform_scan: bool

    # Current action being taken
    current_action: Optional[str]
    
    # Track closed trades for the daily reviewer
    trades_today: List[dict]
    
    # Error tracking
    last_error: Optional[str]


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

async def scanner_node(state: TradingState) -> dict:
    """
    Scans for new trading signals.
    This runs periodically to find SMA crossover opportunities.
    """
    # Only scan if the flag is set
    if not state.get("perform_scan", False):
        return {
            "signals": state.get("signals", []),
            "current_action": "monitoring_positions",
            "last_error": None
        }

    from trading_bot.scanner import scan_for_signals
    from dataclasses import asdict
    
    try:
        signals = await scan_for_signals()
        # Convert dataclasses to dicts for serialization
        signals_dict = [asdict(s) for s in signals]
        
        # Limit to top 10 signals to avoid recursion limits and over-trading
        signals_dict = signals_dict[:10]
        
        return {
            "signals": signals_dict,
            "current_action": "scanned_for_signals",
            "last_error": None
        }
    except Exception as e:
        return {
            "signals": [],
            "current_action": "scan_failed",
            "last_error": str(e)
        }


async def account_sync_node(state: TradingState) -> dict:
    """
    Syncs account information and positions from Alpaca.
    """
    import os
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import OrderStatus
    
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    # Default to paper if not specified
    paper = True 
    
    # Simple direct client for sync
    client = TradingClient(api_key, secret_key, paper=True)
    
    try:
        account = client.get_account()
        
        # Get positions
        alpaca_positions = client.get_all_positions()
        
        # Convert to list of dicts
        positions = []
        for p in alpaca_positions:
            positions.append({
                "symbol": p.symbol,
                "side": p.side,  # 'long' or 'short'
                "quantity": float(p.qty),
                "entry_price": float(p.avg_entry_price),
                "entry_time": datetime.now(), # Placeholder
                "stop_loss": 0.0, # Placeholder
                "take_profit": 0.0, # Placeholder
                "order_id": ""
            })
            
        # Get today's closed/filled orders for the reviewer
        today_str = datetime.now().strftime("%Y-%m-%d")
        # Note: listing orders by date requires formatted request in newer SDKs, 
        # but for simplicity we will get last 50 closed and filter in python if needed
        # or just rely on the fact that we run this daily.
        
        request_params = GetOrdersRequest(
            status=OrderStatus.CLOSED,
            limit=10,
            nested=True
        )
        closed_orders = client.get_orders(filter=request_params)
        
        # Filter for today (rudimentary check, assuming local time matches approx)
        trades_today = []
        for o in closed_orders:
            # Check if filled_at is today
            if o.filled_at and o.filled_at.strftime("%Y-%m-%d") == today_str:
                trades_today.append({
                    "symbol": o.symbol,
                    "side": o.side,
                    "qty": float(o.qty or 0),
                    "entry_price": float(o.filled_avg_price or 0),
                    "pnl": 0.0, # Alpaca orders don't store PnL directly, would need to calc
                    "status": o.status
                })

        # Calculate rough PnL for closed trades (approximated)
        # In a real system, we'd match buy/sell pairs or use account.equity change
        # For now, we trust the daily_pnl from account matches these trades.

        return {
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "daily_pnl": float(account.equity) - float(account.last_equity),
            "positions": positions,
            "trades_today": trades_today,
            "current_action": "synced_account"
        }
    except Exception as e:
        print(f"   ⚠️  Sync Warning: Could not refresh account data: {e}")
        return {
            "current_action": "sync_failed",
            "last_error": str(e)
        }


async def agent_node(state: TradingState, trading_config: TradingConfig, model: ChatGoogleGenerativeAI, tools: list) -> dict:
    """
    The main trading agent - makes decisions about what to do.
    """
    # Bind tools to model
    model_with_tools = model.bind_tools(tools)
    
    # Build context message
    # Get Memory Context
    active_rules = memory_manager.get_rules_text()
    recent_history = memory_manager.get_recent_performance()

    context = f"""
Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}
Market Open: {state.get('is_market_open', False)}
Should Close All: {state.get('should_close_all', False)}
Daily Loss Limit Hit: {state.get('daily_loss_limit_hit', False)}

Account Status:
- Buying Power: ${state.get('buying_power', 0):,.2f}
- Portfolio Value: ${state.get('portfolio_value', 0):,.2f}
- Daily P&L: ${state.get('daily_pnl', 0):,.2f}

Open Positions ({len(state.get('positions', []))}):
{_format_positions(state.get('positions', []))}

New Signals ({len(state.get('signals', []))}):
{_format_signals(state.get('signals', []))}

=========================================
🧠 AGENT MEMORY (LEARNED LESSONS)
=========================================
YOUR RULEBOOK (Evolved from experience):
{active_rules}

RECENT PERFORMANCE:
{recent_history}
=========================================

Based on the above, decide what action to take. You can:
1. Enter new positions (if signals are good and we have capacity)
2. Exit positions (if stop/target hit or should close all)
3. Do nothing (if no good opportunities)

Think step by step about risk management before acting.
"""
    
    messages = state.get("messages", []) + [HumanMessage(content=context)]
    
    response = await model_with_tools.ainvoke(messages)
    
    return {
        "messages": [response],
        "current_action": "agent_decided"
    }


def _format_positions(positions: List[dict]) -> str:
    """Format positions for display."""
    if not positions:
        return "  (none)"
    
    lines = []
    for p in positions:
        # p is a dict now
        lines.append(f"  - {p['symbol']}: {p['quantity']} shares @ ${p['entry_price']:.2f} "
                    f"(Stop: ${p['stop_loss']:.2f}, Target: ${p['take_profit']:.2f})")
    return "\n".join(lines)


def _format_signals(signals: List[dict]) -> str:
    """Format signals for display."""
    if not signals:
        return "  (none)"
    
    lines = []
    for s in signals:
        # s is a dict now
        lines.append(f"  - {s['symbol']} [{s['signal_type']}]: ${s['price']:.2f} "
                    f"(SMA: ${s['sma']:.2f}, Vol Ratio: {s['volume_ratio']:.2f})")
    return "\n".join(lines)


def should_continue(state: TradingState) -> Literal["tools", "end"]:
    """Determine if we should execute tools or end."""
    messages = state.get("messages", [])
    if not messages:
        return "end"
    
    last_message = messages[-1]
    
    # If the last message has tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "end"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

async def create_trading_graph(config: TradingConfig) -> StateGraph:
    """
    Creates the LangGraph trading graph.
    
    The graph flow:
    1. Scanner finds signals
    2. Account sync gets current state
    3. Agent decides what to do
    4. Tools execute the decision
    5. Loop back or end
    """
    
    # Initialize LLM
    model = ChatGoogleGenerativeAI(
        model=config.llm_model,
        temperature=config.llm_temperature,
    )
    
    # Get trading tools (includes MCP tools from Alpaca)
    tools = await get_trading_tools(config)
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Build the graph
    workflow = StateGraph(TradingState)
    
    # Add nodes
    # Add nodes
    from functools import partial
    workflow.add_node("scanner", scanner_node)
    workflow.add_node("account_sync", account_sync_node)
    
    # Use partial to bind arguments to the async function
    agent_node_bound = partial(agent_node, trading_config=config, model=model, tools=tools)
    workflow.add_node("agent", agent_node_bound)
    
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "scanner")
    workflow.add_edge("scanner", "account_sync")
    workflow.add_edge("account_sync", "agent")
    
    # Conditional edge from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After tools, go back to agent (for multi-step reasoning)
    workflow.add_edge("tools", "agent")
    
    # Compile with memory
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    return graph
