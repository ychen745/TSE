import backtrader as bt
import backtrader.indicators as btind

class BbBreakout(bt.Strategy):
    def __init__(self):
        self.bb = btind.BollingerBands(period=20, devfactor=2)

    def next(self):
        if self.data.close[0] > self.bb.lines.top[0]:
            self.buy()
        elif self.data.close[0] < self.bb.lines.bot[0]:
            self.sell()
