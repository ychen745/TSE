import backtrader as bt
import backtrader.indicators as btind

class RsiReversion(bt.Strategy):
    def __init__(self):
        self.rsi = bt.ind.RSI(period=14)
        # self.rsi = btind.RSI(period=14)

    def next(self):
        if self.rsi < 30:
            self.buy(size=0.5)
        elif self.rsi > 50:
            self.sell()
