
import asyncio
import os
from dotenv import load_dotenv
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from trading_bot.scanner import scan_for_signals

async def main():
    print("Loading environment...")
    load_dotenv()
    
    api_key = os.getenv("ALPACA_API_KEY")
    if not api_key:
        print("❌ ALPACA_API_KEY not found in environment")
        return

    print("Starting scan...")
    try:
        signals = await scan_for_signals()
        print(f"\nScan complete. Found {len(signals)} signals.")
        for s in signals:
            print(f" - {s.symbol}: {s.signal_type} @ {s.price}")
    except Exception as e:
        print(f"❌ Scan failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
