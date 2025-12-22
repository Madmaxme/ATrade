# ATrade

A LangGraph-powered autonomous day trading agent that uses SMA crossover signals and executes trades via Alpaca.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          ATRADE                             │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   EVENT LOOP                         │   │
│  │         (Runs continuously 9:30 AM - 4 PM ET)       │   │
│  └─────────────────────────────────────────────────────┘   │
│           │              │              │                   │
│           ▼              ▼              ▼                   │
│  ┌──────────────┐ ┌─────────────┐ ┌──────────────┐         │
│  │   SCANNER    │ │  POSITION   │ │    RISK      │         │
│  │  SMA signals │ │  MANAGER    │ │   MANAGER    │         │
│  │  every 5 min │ │  Trail stops│ │  Max loss    │         │
│  └──────────────┘ │  Take profit│ │  Position sz │         │
│                   └─────────────┘ └──────────────┘         │
│           │              │              │                   │
│           └──────────────┼──────────────┘                   │
│                          ▼                                  │
│                 ┌──────────────┐                            │
│                 │    AGENT     │                            │
│                 │   (Gemini)   │                            │
│                 │  LangGraph   │                            │
│                 └──────────────┘                            │
│                          │                                  │
│                          ▼                                  │
│                 ┌──────────────┐                            │
│                 │  ALPACA MCP  │                            │
│                 │   Executor   │                            │
│                 └──────────────┘                            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 3:55 PM AUTO-CLOSE                   │   │
│  │            Flatten all positions                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **SMA Crossover Scanner** - Scans S&P 500 for 21-day SMA crossovers
- **LangGraph Agent** - Gemini-powered decision making with tool use
- **Alpaca MCP Integration** - Executes trades via Model Context Protocol
- **Risk Management** - Position sizing, stop losses, daily loss limits
- **Day Trading Rules** - Closes all positions before market close
- **Dockerized** - Easy deployment and scaling

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)
- Google API key
- Alpaca paper trading account

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd trading_bot_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Install uv (for Alpaca MCP)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Edit `trading_bot/config.py` or set environment variables:

```python
# Risk Management
MAX_POSITION_SIZE = 0.10      # 10% max per trade
MAX_POSITIONS = 5             # Max concurrent positions
STOP_LOSS_PCT = 0.02          # 2% stop loss
TAKE_PROFIT_PCT = 0.04        # 4% take profit (2:1 R/R)
MAX_DAILY_LOSS_PCT = 0.02     # 2% max daily loss
```

### Running

```bash
# Direct run
python main.py

# Or with Docker
docker-compose up --build
```

## Strategy

The bot uses a simple but effective day trading strategy:

1. **Signal Generation**: Scans S&P 500 for stocks crossing above/below their 21-day SMA
2. **Signal Filtering**: 
   - Volume ratio > 0.5 (conviction check)
   - Price within 5% of SMA (not chasing)
   - Stock price $10-$500 (liquidity)
3. **Entry**: Market/limit orders based on signal quality
4. **Exit**: 
   - Stop loss at 2%
   - Take profit at 4%
   - Close all by 3:55 PM ET

## Risk Management

**Hard-coded rules the bot NEVER violates:**

| Rule | Value | Why |
|------|-------|-----|
| Max Position Size | 10% | Diversification |
| Max Positions | 5 | Concentration risk |
| Stop Loss | 2% | Limit downside |
| Take Profit | 4% | 2:1 reward/risk |
| Daily Loss Limit | 2% | Live to trade another day |
| EOD Close | 3:55 PM | Day trading rules |

## Monitoring

The bot logs all decisions and trades:

```
logs/
├── trades_20251217.csv      # All executed trades
├── signals_20251217.csv     # All signals generated
└── decisions_20251217.log   # Agent reasoning
```

## Development

### Project Structure

```
trading_bot_project/
├── main.py                 # Entry point
├── trading_bot/
│   ├── __init__.py
│   ├── config.py           # Configuration
│   ├── graph.py            # LangGraph definition
│   ├── tools.py            # Agent tools + MCP
│   ├── scanner.py          # SMA signal scanner
│   ├── scheduler.py        # Market hours scheduler
│   └── prompts.py          # Agent prompts
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

### Adding New Strategies

1. Create a new scanner in `trading_bot/scanner.py`
2. Add evaluation tool in `trading_bot/tools.py`
3. Update prompts in `trading_bot/prompts.py`

## Disclaimer

⚠️ **This is for educational purposes only.**

- Always use paper trading first
- Past performance doesn't guarantee future results
- Day trading involves significant risk
- Never risk money you can't afford to lose

## License

MIT
# ATrade
