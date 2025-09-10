import backtrader as bt

class MacdCross(bt.Strategy):
    def __init__(self):
        self.macd = bt.ind.MACD()
        self.signal = bt.ind.MACD().signal

    def next(self):
        if self.macd[0] > self.signal[0] and not self.position:
            self.buy()
        elif self.macd[0] < self.signal[0] and self.position:
            self.sell()
