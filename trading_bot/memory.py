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
    
    # The 'Action' (What strategy/config we used)
    config_used: Dict[str, Any]
    champion_stock: str
    system_version: str = "1.0" # Default for legacy episodes
    
    # The 'Reward' (Outcome)
    start_equity: float
    end_equity: float
    pnl: float
    pnl_pct: float
    win: bool
    
    # Quant Lab Data (New)
    optimization_data: Optional[Dict] = None # Stores {sma: 10, stop: 0.07, source: "QuantLab"}
    
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
                    # Filter out any extra keys from old memory files
                    valid_keys = DailyEpisode.__match_args__ if hasattr(DailyEpisode, "__match_args__") else DailyEpisode.__annotations__.keys()
                    # Simple sanitization
                    clean_episodes = []
                    for item in data:
                        clean_item = {k: v for k, v in item.items() if k in DailyEpisode.__annotations__}
                        clean_episodes.append(DailyEpisode(**clean_item))
                    self.episodes = clean_episodes
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
            summary += f"- {ep.date}: {emoji} {ep.champion_stock} ({ep.pnl_pct:+.2f}%)\n"
        
        return summary

    def get_learning_context(self) -> str:
        """
        Analyze memory to find patterns, specifically regarding version evolution.
        """
        if not self.episodes:
            return "No past trading episodes found. This is a fresh system."
            
        # Group by 'Strategy Version' (the part before the first dash)
        # e.g. "1.1-rev-abc" -> "1.1"
        def get_strat(v_str): return v_str.split('-')[0]
        
        current_strat = get_strat(self.episodes[-1].system_version)
        
        v_episodes = [ep for ep in self.episodes if get_strat(ep.system_version) == current_strat]
        legacy_episodes = [ep for ep in self.episodes if get_strat(ep.system_version) != current_strat]
        
        v_wins = [ep for ep in v_episodes if ep.win]
        v_win_rate = (len(v_wins) / len(v_episodes) * 100) if v_episodes else 0
        
        legacy_win_rate = 0
        if legacy_episodes:
            legacy_wins = [ep for ep in legacy_episodes if ep.win]
            legacy_win_rate = len(legacy_wins) / len(legacy_episodes) * 100

        summary = f"""
🧠 MEMORY INSIGHTS:
- Current Strategy Logic: v{current_strat}
- Performance on this logic: {v_win_rate:.1f}% win rate over {len(v_episodes)} days.
- Legacy Performance (Older Logic): {legacy_win_rate:.1f}% win rate over {len(legacy_episodes)} days.
"""
        if v_win_rate > legacy_win_rate and legacy_episodes:
            summary += "- ANALYSIS: Current version is OUTPERFORMING legacy code. Keep current configuration.\n"
        elif v_win_rate < legacy_win_rate and legacy_episodes:
            summary += "- ANALYSIS: Current version is UNDERPERFORMING legacy code. Consider reverting major changes.\n"
            
        return summary
