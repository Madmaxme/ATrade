"""
Trading Bot Tools
==================
Tools available to the trading agent, including Alpaca MCP integration.
"""

from typing import List
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from trading_bot.config import TradingConfig


# =============================================================================
# MCP CLIENT SETUP
# =============================================================================

_mcp_client = None


async def get_mcp_client(config: TradingConfig) -> MultiServerMCPClient:
    """Get or create the MCP client for Alpaca."""
    global _mcp_client
    
    if _mcp_client is None:
        # Check if alpaca-mcp-server is installed directly (e.g. via pip in Docker)
        import shutil
        executable = shutil.which("alpaca-mcp-server")
        
        if executable:
            command = "alpaca-mcp-server"
            args = ["serve"]
            print("   ✓ Local trading engine found")
        else:
            # Fallback to uvx (slower, requires internet)
            command = "uvx"
            args = ["alpaca-mcp-server", "serve"]
            print("   ℹ️  Initializing cloud trading engine (uvx)")

        _mcp_client = MultiServerMCPClient({
            "alpaca": {
                "command": command,
                "args": args,
                "transport": "stdio",
                "env": {
                    "ALPACA_API_KEY": config.alpaca_api_key,
                    "ALPACA_SECRET_KEY": config.alpaca_secret_key,
                }
            }
        })
    
    return _mcp_client


# =============================================================================
# LOCAL TOOLS (Non-MCP)
# =============================================================================

@tool
def calculate_position_size(
    portfolio_value: float,
    max_position_pct: float,
    stock_price: float,
    stop_loss_pct: float
) -> dict:
    """
    Calculate the appropriate position size based on risk management rules.
    
    Args:
        portfolio_value: Total portfolio value
        max_position_pct: Maximum percentage of portfolio per position
        stock_price: Current stock price
        stop_loss_pct: Stop loss percentage (e.g., 0.02 for 2%)
    
    Returns:
        Dictionary with shares to buy and dollar amount
    """
    max_dollars = portfolio_value * max_position_pct
    shares = int(max_dollars / stock_price)
    actual_dollars = shares * stock_price
    risk_per_share = stock_price * stop_loss_pct
    total_risk = shares * risk_per_share
    
    return {
        "shares": shares,
        "dollar_amount": round(actual_dollars, 2),
        "risk_per_share": round(risk_per_share, 2),
        "total_risk": round(total_risk, 2),
        "pct_of_portfolio": round((actual_dollars / portfolio_value) * 100, 2)
    }


@tool
def calculate_stop_and_target(
    entry_price: float,
    side: str,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04
) -> dict:
    """
    Calculate stop loss and take profit prices for a trade.
    
    Args:
        entry_price: The price at which we're entering
        side: 'buy' for long, 'sell' for short
        stop_loss_pct: Stop loss percentage (default 2%)
        take_profit_pct: Take profit percentage (default 4%)
    
    Returns:
        Dictionary with stop_loss and take_profit prices
    """
    if side.lower() == "buy":
        stop_loss = entry_price * (1 - stop_loss_pct)
        take_profit = entry_price * (1 + take_profit_pct)
    else:  # short
        stop_loss = entry_price * (1 + stop_loss_pct)
        take_profit = entry_price * (1 - take_profit_pct)
    
    return {
        "entry_price": entry_price,
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "risk_reward_ratio": round(take_profit_pct / stop_loss_pct, 2)
    }


@tool
def check_daily_loss_limit(
    daily_pnl: float,
    portfolio_value: float,
    max_daily_loss_pct: float = 0.02
) -> dict:
    """
    Check if daily loss limit has been hit.
    
    Args:
        daily_pnl: Today's P&L (negative if losing)
        portfolio_value: Total portfolio value
        max_daily_loss_pct: Maximum daily loss percentage (default 2%)
    
    Returns:
        Dictionary indicating if limit hit and details
    """
    max_loss_dollars = portfolio_value * max_daily_loss_pct
    limit_hit = daily_pnl <= -max_loss_dollars
    
    return {
        "limit_hit": limit_hit,
        "daily_pnl": round(daily_pnl, 2),
        "max_allowed_loss": round(-max_loss_dollars, 2),
        "remaining_before_limit": round(max_loss_dollars + daily_pnl, 2) if daily_pnl < 0 else round(max_loss_dollars, 2),
        "message": "⚠️ DAILY LOSS LIMIT HIT - STOP TRADING" if limit_hit else "Within daily loss limits"
    }


@tool
def evaluate_signal_quality(
    signal_type: str,
    pct_from_sma: float,
    volume_ratio: float,
    daily_change_pct: float,
    min_volume_ratio: float = 0.5,
    max_pct_from_sma: float = 5.0
) -> dict:
    """
    Evaluate the quality of a trading signal.
    
    Args:
        signal_type: 'BUY' or 'SELL'
        pct_from_sma: Percentage distance from SMA
        volume_ratio: Today's volume / average volume
        daily_change_pct: Today's price change percentage
        min_volume_ratio: Minimum volume ratio to consider
        max_pct_from_sma: Maximum % from SMA (avoid chasing)
    
    Returns:
        Quality assessment with score and reasoning
    """
    score = 0
    reasons = []
    warnings = []
    
    # Volume check
    if volume_ratio >= min_volume_ratio:
        score += 1
        reasons.append(f"Decent volume ({volume_ratio:.2f}x avg)")
    else:
        warnings.append(f"Low volume ({volume_ratio:.2f}x avg)")
    
    if volume_ratio >= 1.5:
        score += 1
        reasons.append("Strong volume conviction")
    
    # Distance from SMA
    abs_pct = abs(pct_from_sma)
    if abs_pct <= 2.0:
        score += 1
        reasons.append(f"Close to SMA ({pct_from_sma:.1f}%)")
    elif abs_pct > max_pct_from_sma:
        warnings.append(f"Far from SMA ({pct_from_sma:.1f}%) - might be chasing")
    
    # Direction alignment
    if signal_type == "BUY" and daily_change_pct > 0:
        score += 1
        reasons.append("Momentum aligned (up day)")
    elif signal_type == "SELL" and daily_change_pct < 0:
        score += 1
        reasons.append("Momentum aligned (down day)")
    
    # Determine quality
    if score >= 3:
        quality = "HIGH"
    elif score >= 2:
        quality = "MEDIUM"
    else:
        quality = "LOW"
    
    return {
        "quality": quality,
        "score": score,
        "max_score": 4,
        "reasons": reasons,
        "warnings": warnings,
        "recommendation": "TRADE" if quality in ["HIGH", "MEDIUM"] else "SKIP"
    }


@tool 
def format_trade_log(
    action: str,
    symbol: str,
    quantity: int,
    price: float,
    reason: str
) -> str:
    """
    Format a trade action for logging.
    
    Args:
        action: 'BUY', 'SELL', 'STOP_OUT', 'TAKE_PROFIT', etc.
        symbol: Stock symbol
        quantity: Number of shares
        price: Execution price
        reason: Why the trade was made
    
    Returns:
        Formatted log string
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = quantity * price
    
    return f"[{timestamp}] {action} {quantity} {symbol} @ ${price:.2f} (${total:,.2f}) - {reason}"


@tool
def get_market_sentiment(symbol: str) -> dict:
    """
    Get market sentiment and news for a specific symbol using Tavily.
    
    Args:
        symbol: Stock symbol (e.g. 'AAPL')
    
    Returns:
        Dictionary with news summary and headlines
    """
    try:
        from tavily import TavilyClient
        import os
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {"error": "TAVILY_API_KEY not set", "sentiment": "unknown"}
            
        client = TavilyClient(api_key=api_key)
        
        # Search specifically for news
        query = f"{symbol} stock news today why is it moving"
        response = client.search(query, topic="news", time_range="day", max_results=3)
        
        results = []
        for r in response.get("results", []):
            results.append(f"- {r['title']}: {r['content'][:200]}...")
            
        return {
            "symbol": symbol,
            "headline_count": len(results),
            "top_news": results,
            "sentiment": "See news content for details"
        }
    except Exception as e:
        return {"error": str(e), "sentiment": "unknown"}


# =============================================================================
# GET ALL TOOLS
# =============================================================================

async def get_trading_tools(config: TradingConfig) -> List:
    """
    Get all tools available to the trading agent.
    Combines local tools with Alpaca MCP tools.
    """
    # Local tools
    local_tools = [
        calculate_position_size,
        calculate_stop_and_target,
        check_daily_loss_limit,
        evaluate_signal_quality,
        get_market_sentiment,
        format_trade_log,
    ]
    
    # Get MCP tools from Alpaca
    try:
        mcp_client = await get_mcp_client(config)
        mcp_tools = await mcp_client.get_tools()
        
        # Combine all tools
        all_tools = local_tools + list(mcp_tools)
        print(f"   ✓ Trading Capability: {len(local_tools)} internal modules + {len(mcp_tools)} exchange connectors active")
        
        return all_tools
        
    except Exception as e:
        print(f"⚠️ Could not load MCP tools: {e}")
        print("  Running with local tools only (no trade execution)")
        return local_tools
