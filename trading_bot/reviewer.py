"""
Daily Reviewer
===============
The 'Teacher' module that runs after market close to analyze performance
and update the agent's rulebook.
"""

from typing import List, Dict
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from trading_bot.config import TradingConfig
from trading_bot.memory import memory_manager, TradeEpisode

REVIEW_SYSTEM_PROMPT = """You are a Master Trading Coach. Your job is to analyze the daily performance of a junior trading bot ("The Sniper") and update its Rulebook to improve future performance.

You will be given:
1. The trade made today (The "Champion" trade)
2. The outcome (PnL)
3. The Junior's reasoning for taking the trade
4. Validated market data (was the market actually bullish/bearish?)
5. The Current Rulebook

Your Goal:
Determine WHY the trade succeeded or failed.
- If it failed: Was it bad luck, or a flaw in strategy? (e.g., entered too early, ignored volume).
- If it succeeded: Reinforce the behavior.

Output TWO things:
1. A brief "Reflection" on the specific trade (Journal Entry).
2. An UPDATED List of Rules. You can modify, delete, or add rules. KEEP THE LIST CONCISE (Max 10 rules).
"""

async def run_daily_review(
    config: TradingConfig,
    todays_trade: Dict, 
    daily_pnl: float,
    current_state: Dict
):
    """
    Analyzes the day's single trade and updates the global rulebook.
    """
    if not todays_trade:
        print("   ℹ️  No trades made today. Skipping Daily Review.")
        return

    print("\n👨‍🏫 Starting Daily Review Session...")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=config.llm_model,
        temperature=0.2, # Slightly higher for creative reflection
    )
    
    # Gather Context
    symbol = todays_trade.get("symbol", "Unknown")
    entry_price = todays_trade.get("entry_price", 0.0)
    exit_price = todays_trade.get("exit_price", entry_price) # Fallback if not recorded
    pnl_dollars = todays_trade.get("pnl", 0.0)
    
    # Calculate PnL %
    pnl_pct = 0.0
    if entry_price > 0:
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        if todays_trade.get("side") == "sell": # Short
            pnl_pct = pnl_pct * -1
            
    current_rules = memory_manager.get_rules_text()
    
    prompt = f"""
    REVIEW SESSION: {datetime.now().strftime('%Y-%m-%d')}
    
    1. TODAY'S TRADE:
    - Symbol: {symbol} ({todays_trade.get('side', 'long')})
    - Entry: ${entry_price:.2f}
    - Exit: ${exit_price:.2f}
    - PnL: ${pnl_dollars:.2f} ({pnl_pct:+.2f}%)
    - Result: {"PROFITABLE" if pnl_dollars > 0 else "LOSS"}
    
    2. CURRENT RULES:
    {current_rules}
    
    3. TASK:
    - Write a short reflection on why this result happened.
    - Rewrite the Rulebook. If the trade lost due to a missing rule, ADD IT. If a rule was violated, strengthen it.
    
    RESPONSE FORMAT:
    REFLECTION: <text>
    RULES:
    1. <rule>
    2. <rule>
    ...
    """
    
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content
        
        # Parse Response (Simple parsing)
        reflection_part = content.split("RULES:")[0].replace("REFLECTION:", "").strip()
        rules_part = content.split("RULES:")[1].strip()
        
        new_rules = []
        for line in rules_part.split("\n"):
            clean_line = line.strip()
            # Remove numbering (1. , - , etc)
            if clean_line and (clean_line[0].isdigit() or clean_line.startswith("-")):
                # Find first space
                try:
                    rule_text = clean_line.split(" ", 1)[1]
                    new_rules.append(rule_text)
                except:
                    new_rules.append(clean_line)
                    
        # Update Memory
        episode = TradeEpisode(
            date=datetime.now().strftime('%Y-%m-%d'),
            symbol=symbol,
            signal_type=todays_trade.get('side', 'long').upper(),
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            pnl_dollars=pnl_dollars,
            market_notes="Market data analysis pending integration", # Placeholder
            reflection=reflection_part
        )
        
        memory_manager.add_episode(episode)
        memory_manager.update_rules(new_rules)
        
        print("   ✅ Review Complete.")
        print(f"   📝 Reflection: {reflection_part[:100]}...")
        print(f"   🧠 Logic Updated: {len(new_rules)} active rules.")
        
    except Exception as e:
        print(f"   ⚠️ Review Failed: {e}")
