import backtrader as bt

class SmaCross(bt.Strategy):
    def __init__(self):
        self.sma1 = bt.ind.SMA(period=20)
        self.sma2 = bt.ind.SMA(period=50)

    def next(self):
        if self.sma1 > self.sma2 and not self.position:
            self.buy()
        elif self.sma1 < self.sma2 and self.position:
            self.sell()
