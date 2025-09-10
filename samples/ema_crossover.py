import backtrader as bt

class EmaCross(bt.Strategy):
    def __init__(self):
        self.ema1 = bt.ind.EMA(period=12)
        self.ema2 = bt.ind.EMA(period=26)

    def next(self):
        if self.ema1 > self.ema2 and not self.position:
            self.buy(size=1)
        elif self.ema1 < self.ema2 and self.position:
            self.sell(size=1)
