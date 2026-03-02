<p align="left">

## 1. **lstm_strat.ipynb**
Crypto Portfolio (15 cryptos) - Long short-term memory (LSTM) strategy

### Strategy
LSTM model on crypto data from 2023-07 -> Walk-Forward test, step: 2 months. For optimization Log MDD loss was used instead of mse, rmse, etc.

### Results
* All test periods (2 months each) with a positive ***PnL ranging from 2% to 29%***
* Max drawdown of 20% - w/o stop losses (potentially lower mdd)
* further, detailed results in ***lstm_strat_walk_forward.pdf***

used 15 cryptos, more could be used (1d, 1hr, 15min data availability from 2023 onwards)

## 2. **rank_strat.ipynb**
Percentile-Rank Momentum Strategy (Landolfi 2025)

### Strategy
Vol-normalized returns → separate percentile ranking of positive vs negative price changes → composite scores of price changes → threshold entry/exit (hysteresis) + max-hold cap.

### Results
* Out of Sample ***Sharpe - 1.16***
* ***85% return*** - 2024/01/01 to 2025/12/01
* ***23% MDD***

Tested on ETHUSDT with transaction costs included. Data from Binance API. Similar results for SOL, XRP. Slightly more conservative results for BTCUSDT

## 3. **pairs_trading.ipynb**
#### Statistical Arbitrage on EURO STOXX 50
* Train - 1Y
* Test - 6M (2024 Jan to June and July to Dec)
Sharpe > 1 (positive returns)
</p>
