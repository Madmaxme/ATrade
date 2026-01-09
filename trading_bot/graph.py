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
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage

from trading_bot.config import TradingConfig
from trading_bot.tools import get_trading_tools
from trading_bot.prompts import TRADER_SYSTEM_PROMPT


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
    
    # NEW: Store optimization results for end-of-day memory
    # Format: { "AAPL": {"rsi_threshold": 25, ...}, ... }
    optimization_history: Optional[dict]
    
    # Error tracking
    last_error: Optional[str]


# =============================================================================
# NODE FUNCTIONS
# =============================================================================




async def scanner_node(state: TradingState, trading_config: TradingConfig) -> dict:
    """
    Scans for new trading signals.
    Runs only if:
    1. 'perform_scan' is True
    2. We have capacity for new positions (current < max)
    """
    # 1. Check scan flag
    if not state.get("perform_scan", False):
        return {
            "signals": state.get("signals", []),
            "current_action": "monitoring_positions",
            "last_error": None
        }

    # 2. Check capacity (Smart Scan)
    current_positions = state.get("positions", [])
    if len(current_positions) >= trading_config.max_positions:
        print(f"   🛡️ Smart Scan: Skipping market scan (Portfolio Full: {len(current_positions)}/{trading_config.max_positions})")
        return {
            "signals": state.get("signals", []), # Keep existing signals if any
            "current_action": "monitoring_positions",
            "last_error": None
        }

    # 3. Perform Scan
    from trading_bot.scanner import scan_for_signals
    from dataclasses import asdict
    
    try:
        signals = await scan_for_signals()
        # Convert dataclasses to dicts for serialization
        signals_dict = [asdict(s) for s in signals]
        
        # Limit to top 10 signals
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
    
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    # Default to paper if not specified
    paper = True 
    
    # Simple direct client for sync
    client = TradingClient(api_key, secret_key, paper=True)
    
    from trading_bot.config import DEFAULT_CONFIG
    
    try:
        account = client.get_account()
        
        # Get positions
        alpaca_positions = client.get_all_positions()
        
        # Convert to list of dicts
        positions = []
        for p in alpaca_positions:
            # Parse numbers safely
            qty = float(p.qty)
            entry = float(p.avg_entry_price)
            current = float(p.current_price)
            pl = float(p.unrealized_pl)
            pl_pct = float(p.unrealized_plpc) * 100
            
            # Re-calculate stops based on strategy rules (since we don't store them externally)
            # Long positions only for now
            if p.side == 'long':
                stop = entry * (1 - DEFAULT_CONFIG.stop_loss_pct)
                target = entry * (1 + DEFAULT_CONFIG.take_profit_pct)
            else:
                stop = entry * (1 + DEFAULT_CONFIG.stop_loss_pct)
                target = entry * (1 - DEFAULT_CONFIG.take_profit_pct)

            positions.append({
                "symbol": p.symbol,
                "side": p.side,
                "quantity": qty,
                "entry_price": entry,
                "current_price": current,
                "unrealized_pl": pl,
                "unrealized_pl_pct": pl_pct,
                "stop_loss": stop,
                "take_profit": target,
                "entry_time": datetime.now(), # Placeholder
                "order_id": ""
            })
            
        return {
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "daily_pnl": float(account.equity) - float(account.last_equity),
            "positions": positions,
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
    
    # Filter signals if at capacity to prevent wasting cycles on impossible trades
    # NOTE: With the new tool-based scanning, the agent decides when to scan.
    # However, if there ARE signals in the state (leftover), we might still want to show them.
    scan_signals = state.get('signals', [])
    current_positions = state.get('positions', [])

    # Build context message
    # Use robust time conversion from config
    now_et = trading_config.get_now_et()
    
    
    # helper for signals display
    signals_text = _format_signals(scan_signals)
    
    context = f"""
Current Time: {now_et.strftime('%Y-%m-%d %H:%M:%S ET')}
Market Open: {state.get('is_market_open', False)}
Should Close All: {state.get('should_close_all', False)}
Daily Loss Limit Hit: {state.get('daily_loss_limit_hit', False)}

Account Status:
- Buying Power: ${state.get('buying_power', 0):,.2f}
- Portfolio Value: ${state.get('portfolio_value', 0):,.2f}
- Daily P&L: ${state.get('daily_pnl', 0):,.2f}

Open Positions ({len(current_positions)}):
{_format_positions(current_positions)}

New/Existing Signals (from previous scans):
{signals_text}

Based on the above, decide what action to take. You can:
1. Enter new positions (if signals are good and we have capacity)
2. Exit positions (if stop/target hit or should close all)
3. Do nothing (if no good opportunities)

IMPORTANT:
- If you just decided to HOLD or SKIP, do NOT call `format_trade_log`. Just state your decision.
- Only call `format_trade_log` when you actually EXECUTE a trade (Buy/Sell).

Think step by step about risk management before acting.
"""
    
    # helper to construct messages
    messages_state = state.get("messages", [])
    
    # NEW: Capture Optimization Data from Tool Outputs
    # We scan recent messages for "find_best_settings_tool" outputs
    opt_history = state.get("optimization_history", {}) or {}
    history_updated = False
    
    for msg in messages_state:
        if isinstance(msg, ToolMessage) and msg.name == "find_best_settings_tool":
            content = msg.content
            if "OPTIMIZATION RESULT" in content:
                try:
                    import json
                    # Attempt to parse
                    data = json.loads(content)
                    if isinstance(data, dict) and "data" in data:
                        raw = data["data"]
                        symbol = raw.get("symbol")
                        if symbol:
                            opt_history[symbol] = raw
                            history_updated = True
                except:
                    # Fallback if text format
                    pass
    
    # If we updated history, we must return it in the final dict (LangGraph merges updates)
    # logic is handled at return statement, but we need to pass this state to return
    
    new_messages = []
    
    # Add system prompt if this is the start of conversation
    if not messages_state:
        # Note: TRADER_SYSTEM_PROMPT is imported at top level
        new_messages.append(SystemMessage(content=TRADER_SYSTEM_PROMPT))
    
    # Add the current context as a user message
    # IMPORTANT: We must persist this message to state so the history is valid
    # (previous turns must have User -> Model structure)
    human_msg = HumanMessage(content=context)
    new_messages.append(human_msg)
    
    # Combine for model input
    messages_input = messages_state + new_messages
    
    response = await model_with_tools.ainvoke(messages_input)

    # Debug logging for user transparency
    if response.tool_calls:
        tool_names = [t['name'] for t in response.tool_calls]
        print(f"   🤖 Agent Action: Calling {len(tool_names)} tools: {', '.join(tool_names)}")
    elif response.content:
        # Just a thought/text response
        content_str = ""
        if isinstance(response.content, list):
             # Extract text from list content if it's the standard format
             for block in response.content:
                 if isinstance(block, dict) and block.get("type") == "text":
                     content_str += block.get("text", "")
                 else:
                     content_str += str(block)
        else:
             content_str = str(response.content)
             
        # Print a clean preview
        clean_content = content_str.strip().replace("\n", " ")
        print(f"   🤔 Agent Thought: {clean_content}")
    
    # Check for repetitive log calls to avoid loops
    if response.tool_calls:
        tool_names = [t['name'] for t in response.tool_calls]
        # logic to handle repetitive log calls if needed in future
        pass

    # Return both the new user message(s) and the response to be saved to state
    return {
        "messages": new_messages + [response],
        "current_action": "agent_decided",
        "optimization_history": opt_history # Persist updated history
    }


def _format_positions(positions: List[dict]) -> str:
    """Format positions for display."""
    if not positions:
        return "  (none)"
    
    lines = []
    for p in positions:
        # p is a dict now
        lines.append(f"  - {p['symbol']} ({p['side']}): {p['quantity']} @ ${p['entry_price']:.2f} "
                    f"| Curr: ${p['current_price']:.2f} "
                    f"| P&L: ${p['unrealized_pl']:.2f} ({p['unrealized_pl_pct']:.2f}%) "
                    f"| Stop: ${p['stop_loss']:.2f} | Target: ${p['take_profit']:.2f}")
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

async def create_trading_graph(config: TradingConfig, override_tools: List = None) -> StateGraph:
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
    if override_tools:
        tools = override_tools
    else:
        tools = await get_trading_tools(config)
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Build the graph
    workflow = StateGraph(TradingState)
    
    # Add nodes
    from functools import partial
    
    # Bind config to scanner
    scanner_node_bound = partial(scanner_node, trading_config=config)
    workflow.add_node("scanner", scanner_node_bound)
    
    workflow.add_node("account_sync", account_sync_node)
    
    # Use partial to bind arguments to the async function
    agent_node_bound = partial(agent_node, trading_config=config, model=model, tools=tools)
    workflow.add_node("agent", agent_node_bound)
    
    workflow.add_node("tools", tool_node)
    
    # Add edges
    # Reordered: Sync -> Scanner -> Agent
    workflow.add_edge(START, "account_sync")
    workflow.add_edge("account_sync", "scanner")
    workflow.add_edge("scanner", "agent")
    
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
