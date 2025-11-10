import os, importlib
try:
    mt5 = importlib.import_module("MetaTrader5")
except Exception:
    mt5 = None

def init():
    if mt5 is None:
        raise RuntimeError("MetaTrader5 is not installed. Optional feature.")
    path = os.getenv("MT5_PATH")
    ok = mt5.initialize(path=path) if path else mt5.initialize()
    if not ok:
        raise RuntimeError("MT5 init failed. Set MT5_PATH or open/login MT5 terminal.")

def latest_close(symbol: str, tf=None):
    if mt5 is None: return None
    if tf is None: tf = mt5.TIMEFRAME_M1
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, 2)
    return rates[-1].close if rates is not None else None
