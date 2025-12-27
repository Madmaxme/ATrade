"""
Trading Memory Module
======================
Manages long-term memory for the trading agent, including:
1. Episode History (Past trades and outcomes)
2. Rulebook (Evolving strategy guidelines)
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

MEMORY_FILE = "trading_memory.json"

@dataclass
class TradeEpisode:
    """Represents a single day's 'Champion' trade and its outcome."""
    date: str
    symbol: str
    signal_type: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_dollars: float
    market_notes: str
    reflection: str

class TradingMemory:
    def __init__(self, filepath: str = MEMORY_FILE):
        self.filepath = filepath
        self.episodes: List[dict] = []
        self.rules: List[str] = [
            "Always follow the trend of the S&P 500.",
            "Never trade against a earnings release.",
            "Cut losses quickly, let winners run."
        ]
        self._load()
    
    def _load(self):
        """Load memory from JSON file."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.episodes = data.get("episodes", [])
                    self.rules = data.get("rules", self.rules)
            except Exception as e:
                print(f"⚠️ Failed to load memory: {e}")
    
    def save(self):
        """Save memory to JSON file."""
        data = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rules": self.rules,
            "episodes": self.episodes
        }
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def add_episode(self, episode: TradeEpisode):
        """Record a finished trading episode."""
        self.episodes.append(asdict(episode))
        self.save()
        
    def update_rules(self, new_rules: List[str]):
        """Update the agent's rulebook."""
        self.rules = new_rules
        self.save()
        
    def get_rules_text(self) -> str:
        """Get formatted rules for LLM context."""
        return "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(self.rules)])
    
    def get_recent_performance(self, n: int = 5) -> str:
        """Get text summary of last N trades."""
        if not self.episodes:
            return "No past trading history available."
            
        recent = self.episodes[-n:]
        summary = []
        for ep in recent:
            summary.append(f"- {ep['date']}: {ep['symbol']} ({ep['signal_type']}) -> {ep['pnl_dollars']:+.2f} ({ep['pnl_pct']:+.2f}%)")
        
        return "\n".join(summary)

# Global instance
memory_manager = TradingMemory()
