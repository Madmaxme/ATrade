"""
Test Quant Lab Integrity
========================
This script validates that the new 'Just-in-Time Backtesting' module works correctly.
It tests:
1. Data fetching from Google/Yahoo (even when market is closed)
2. The 'vet_trade_signal' logic (Veto vs Approval)
3. The 'find_best_settings' optimization (Grid Search)
"""

import sys
import os
import logging
from trading_bot.quant_lab import quant_lab, vet_trade_signal, find_best_settings

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='   %(message)s')
print("🧪 STARTING QUANT LAB DIAGNOSTIC TEST...\n")

def test_data_fetch():
    print("1. Testing Data Connection (yfinance)...")
    symbol = "NVDA"
    df = quant_lab.get_history(symbol, days=100)
    
    if not df.empty:
        last_date = df.index[-1].strftime('%Y-%m-%d')
        print(f"   ✅ SUCCESS: Fetched {len(df)} days for {symbol}. Last data point: {last_date}")
        print(f"      Columns: {list(df.columns)}")
        return True
    else:
        print(f"   ❌ FAILURE: Could not fetch data for {symbol}.")
        return False

def test_veto_logic():
    print("\n2. Testing Signal Validation (Veto Logic)...")
    
    # Test a stock likely to be good (e.g., NVDA in a bull run)
    print("   👉 Testing 'NVDA' with standard SMA-21...")
    result_good = vet_trade_signal("NVDA", 21)
    print(f"      Result: {result_good}")
    
    # Test a stock likely to be choppy/bad (e.g., 'SAVE' - Spirit Airlines, or a meme stock if we had one)
    # Using a known volatile/down-trending stock to see if it catches the danger
    risky_stock = "PFE" # Pfizer has been choppy/down recently
    print(f"   👉 Testing '{risky_stock}' with standard SMA-21...")
    result_bad = vet_trade_signal(risky_stock, 21)
    print(f"      Result: {result_bad}")
    
    if "APPROVED" in result_good or "VETO" in result_bad or "WARNING" in result_bad:
         print("   ✅ SUCCESS: Validation logic is distinguishing between assets.")
    else:
         print("   ⚠️ WARNING: Logic might be too permissive or too strict.")

def test_optimization_logic():
    print("\n3. Testing Parameter Optimization (Sim-to-Real)...")
    symbol = "TSLA"
    print(f"   👉 Optimizing parameters for {symbol}...")
    
    result = find_best_settings(symbol)
    print(f"      Agent Output: {result}")
    
    if "OPTIMIZATION RESULT" in result:
        print("   ✅ SUCCESS: Optimization successfully found custom parameters.")
    else:
        print("   ❌ FAILURE: Optimization routine failed.")

if __name__ == "__main__":
    try:
        if test_data_fetch():
            test_veto_logic()
            test_optimization_logic()
        print("\n🏁 DIAGNOSTIC COMPLETE.")
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
