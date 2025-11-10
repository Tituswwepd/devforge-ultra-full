import ccxt
def price(symbol: str = 'BTC/USDT'):
    binance = ccxt.binance()
    t = binance.fetch_ticker(symbol)
    return {"symbol": symbol, "last": t['last'], "bid": t['bid'], "ask": t['ask']}
