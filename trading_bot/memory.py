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
    
    # The 'Reward' (Outcome)
    start_equity: float
    end_equity: float
    pnl: float
    pnl_pct: float
    win: bool
    
    # Versioning & Extra Data (with defaults at the END)
    system_version: str = "1.0"
    optimization_data: Optional[Dict] = None 
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
                # print(f"   🧠 Memory Loaded: {len(self.episodes)} past trading days.")
            except Exception as e:
                print(f"   ⚠️ Memory Corruption: Could not load {self.filepath}: {e}")
                self.episodes = []
        else:
            # print("   🧠 New Memory Created.")
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
        """Add a new day's experience to the memory. Overwrites only if date AND version match."""
        # Remove existing entry for same date AND same version if exists
        self.episodes = [ep for ep in self.episodes if not (ep.date == episode.date and ep.system_version == episode.system_version)]
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
            note_snip = f" | Note: {ep.notes[:100]}..." if ep.notes and ep.notes != "(Auto-generated)" else ""
            summary += f"- {ep.date}: {emoji} {ep.champion_stock} ({ep.pnl_pct:+.2f}%){note_snip}\n"
        
        return summary

    def get_learning_context(self, current_version: str = None) -> str:
        """
        Analyze memory to find patterns, specifically regarding version evolution.
        """
        if not self.episodes:
            return "No past trading episodes found. This is a fresh system."
            
        def get_strat(v_str): 
            if not v_str or '-' not in v_str: return v_str or "unknown"
            # Captures 'logic-xxxx' from 'logic-xxxx-rev-yyyy'
            parts = v_str.split('-')
            return '-'.join(parts[:2])
        
        # Use the provided version or fall back to the last recorded one
        current_strat = get_strat(current_version) if current_version else get_strat(self.episodes[-1].system_version)
        
        v_episodes = [ep for ep in self.episodes if get_strat(ep.system_version) == current_strat]
        legacy_episodes = [ep for ep in self.episodes if get_strat(ep.system_version) != current_strat]
        
        v_wins = [ep for ep in v_episodes if ep.win]
        v_win_rate = (len(v_wins) / len(v_episodes) * 100) if v_episodes else 0
        
        legacy_win_rate = 0
        if legacy_episodes:
            legacy_wins = [ep for ep in legacy_episodes if ep.win]
            legacy_win_rate = len(legacy_wins) / len(legacy_episodes) * 100

        # Build detailed history list for current version
        v_history = ""
        for ep in v_episodes[-3:]: # Last 3 days of current logic
            note = f" | Note: {ep.notes[:100]}" if ep.notes and ep.notes != "(Auto-generated)" else ""
            v_history += f"  * {ep.date}: {ep.pnl_pct:+.2f}% {note}\n"

        summary = f"""
🧠 MEMORY INSIGHTS:
- Current Strategy Logic: {current_strat}
- Performance on this logic: {v_win_rate:.1f}% win rate over {len(v_episodes)} days.
- Recent Days (This Logic):
{v_history}
- Legacy Performance (Older Logic): {legacy_win_rate:.1f}% win rate over {len(legacy_episodes)} days.
"""
        if v_win_rate > legacy_win_rate and legacy_episodes:
            summary += "- ANALYSIS: Current version is OUTPERFORMING legacy code. Keep current configuration.\n"
        elif v_win_rate < legacy_win_rate and legacy_episodes:
            summary += "- ANALYSIS: Current version is UNDERPERFORMING legacy code. Consider reverting major changes.\n"
            
        return summary
