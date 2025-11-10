from textwrap import dedent

def gen_pine_indicator(symbol: str, timeframe: str, features):
    feats = " // ".join(features)
    return dedent(f"""
//@version=5
indicator(title="AI Ultra — {symbol} {timeframe}", shorttitle="AI-ULTRA", overlay=true)
len = input.int(14)
src = close
ema = ta.ema(src, len)
// Features: {feats}
longCond = ta.crossover(src, ema)
shortCond = ta.crossunder(src, ema)
plot(ema, title="EMA")
plotshape(longCond, title="BUY", style=shape.labelup, text="BUY", location=location.belowbar)
plotshape(shortCond, title="SELL", style=shape.labeldown, text="SELL", location=location.abovebar)
""")

def gen_mt5_bot(symbol: str, strategy: str):
    return dedent(f"""
# MT5 Bot — {symbol} ({strategy})
# Requires MetaTrader5 python package and terminal
import time
try:
    import MetaTrader5 as mt5
except ImportError:
    raise SystemExit("MetaTrader5 not installed. Install it or skip MT5 features.")
if not mt5.initialize():
    raise SystemExit("MT5 init failed")
SYMBOL = "{symbol}"
TF = mt5.TIMEFRAME_M1
def signal():
    rates = mt5.copy_rates_from_pos(SYMBOL, TF, 0, 100)
    closes = [r.close for r in rates]
    ma = sum(closes[-14:]) / 14
    if closes[-1] > ma: return "BUY"
    if closes[-1] < ma: return "SELL"
    return "HOLD"
while True:
    print("Signal:", signal()); time.sleep(5)
""")

def gen_deriv_xml(logic: str):
    return f"""
<xml xmlns="https://developers.google.com/blockly/xml">
  <block type="trade_definition" id="defs" x="90" y="40">
    <field name="MARKET">synthetic_index</field>
    <field name="SYMBOL">R_100</field>
    <field name="TRADETYPE">risefall</field>
  </block>
  <!-- {logic} -->
</xml>
"""
