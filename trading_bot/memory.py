"""
Trading Bot Memory & Learning System
=====================================
Manages the structured memory (JSON) for the bot to learn from past trade outcomes.
This is the KEY COMPONENT for Reinforcement Learning / Self-Optimization.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# File where the "Brain" is stored
MEMORY_FILE = "trading_memory.json"

@dataclass
class DailyEpisode:
    """Represents a single day of trading experience (One 'Episode')."""
    date: str
    
    # The 'State' (What the market looked like)
    market_sentiment: str  # derived from LLM analysis or VIX
    
    # The 'Action' (What strategy/config we used)
    config_used: Dict[str, Any]
    champion_stock: str
    
    # The 'Reward' (Outcome)
    start_equity: float
    end_equity: float
    pnl: float
    pnl_pct: float
    win: bool
    
    # The 'Reflection' (Why did this happen?)
    notes: str = ""


class TradingMemory:
    """Manager for the bot's long-term memory."""
    
    def __init__(self, data_dir: str = "."):
        # Store memory file in the specified data directory
        self.filepath = os.path.join(data_dir, MEMORY_FILE)
        self.episodes: List[DailyEpisode] = []
        self._load_memory()

    def _load_memory(self):
        """Load memory from disk."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.episodes = [DailyEpisode(**item) for item in data]
                print(f"   🧠 Memory Loaded: {len(self.episodes)} past trading days.")
            except Exception as e:
                print(f"   ⚠️ Memory Corruption: Could not load {self.filepath}: {e}")
                self.episodes = []
        else:
            print("   🧠 New Memory Created.")
            self.episodes = []

    def save_memory(self):
        """Save memory to disk."""
        try:
            with open(self.filepath, 'w') as f:
                # Convert dataclasses to dicts
                data = [asdict(ep) for ep in self.episodes]
                json.dump(data, f, indent=2)
            print("   💾 Memory Saved.")
        except Exception as e:
            print(f"   ❌ Failed to save memory: {e}")

    def record_episode(self, episode: DailyEpisode):
        """Add a new day's experience to the memory."""
        # Remove existing entry for same date if exists (overwrite)
        self.episodes = [ep for ep in self.episodes if ep.date != episode.date]
        self.episodes.append(episode)
        self.save_memory()

    def get_recent_performance(self, days: int = 5) -> str:
        """Get a summary of recent performance for the LLM context."""
        recent = sorted(self.episodes, key=lambda x: x.date)[-days:]
        if not recent:
            return "No recent trading history."
            
        summary = "RECENT TRADING HISTORY:\n"
        for ep in recent:
            emoji = "✅" if ep.win else "❌"
            summary += f"- {ep.date}: {emoji} {ep.champion_stock} ({ep.pnl_pct:+.2f}%) | Market: {ep.market_sentiment}\n"
        
        return summary

    def get_learning_context(self) -> str:
        """
        Analyze the entire memory to find patterns. 
        This is the 'Retrieval' part of RAG for the Agent.
        """
        if not self.episodes:
            return ""
            
        wins = [ep for ep in self.episodes if ep.win]
        losses = [ep for ep in self.episodes if not ep.win]
        
        win_rate = len(wins) / len(self.episodes) * 100 if self.episodes else 0
        
        return f"""
MEMORY INSIGHTS:
- Total Days Tracked: {len(self.episodes)}
- Win Rate: {win_rate:.1f}%
- Last 3 Trades: {' '.join(['WIN' if ep.win else 'LOSS' for ep in self.episodes[-3:]])}
"""
