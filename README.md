<p align="left">
  
## **rank_strat.ipynb**
Percentile-Rank Momentum Strategy (Landolfi 2025)

### Strategy
Vol-normalized returns → separate percentile ranking of positive vs negative price changes → composite scores of price changes → threshold entry/exit (hysteresis) + max-hold cap.

### Results
* Out of Sample ***Sharpe - 1.16***
* ***85% return*** - 2024/01/01 to 2025/12/01
* ***23% MDD***

Tested on ETHUSDT with transaction costs included. Data from Binance API. Similar results for SOL, XRP. Slightly more conservative results for BTCUSDT

## **pairs_trading.ipynb**
#### Statistical Arbitrage on EURO STOXX 50
* Train - 1Y
* Test - 6M (2024 Jan to June and July to Dec)
Sharpe > 1 (positive returns)
</p>
