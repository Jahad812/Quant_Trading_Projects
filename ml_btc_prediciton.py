###########################
# IN PROCESS ... #
###########################

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from datetime import datetime, timedelta

def calculate_technical_indicators(df):
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)

    for window in [5, 10, 15, 30]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    return df

def calculate_metrics(portfolio_values):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    total_profit = portfolio_values[-1] / portfolio_values[0]
    
    if len(returns) < 2:
        return total_profit, 0, 0

    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(24 * 60 * 365)
    
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        sortino_ratio = np.mean(returns) / np.std(negative_returns) * np.sqrt(24 * 60 * 365)
    else:
        sortino_ratio = np.inf
    
    return total_profit, sharpe_ratio, sortino_ratio

def run_strategy(df, predictions, weights):
    portfolio_value = [1.0]
    holding = False
    trades = 0
    profitable_trades = 0
    loss_trades = 0
    trading_days = 0 
    
    for i in range(len(df) - max(predictions.keys())):
        current_close = df['Close'].iloc[i]
        
        weighted_log_return_prediction = 0
        for n_ahead, pred_series in predictions.items():
            if i + n_ahead < len(df):
                predicted_close = pred_series.iloc[i]
                log_return_proxy = np.log(predicted_close / current_close)
                weighted_log_return_prediction += weights[n_ahead] * log_return_proxy

        if weighted_log_return_prediction > 0 and not holding:
            holding = True
            trades += 1
            trading_days += 1
            portfolio_value.append(portfolio_value[-1] * (df['Close'].iloc[i+1] / current_close))
        elif weighted_log_return_prediction > 0 and holding:
            trading_days += 1
            portfolio_value.append(portfolio_value[-1] * (df['Close'].iloc[i+1] / current_close))
        elif weighted_log_return_prediction <= 0 and holding:
            holding = False
            portfolio_value.append(portfolio_value[-1] * (df['Close'].iloc[i+1] / current_close))
            if portfolio_value[-1] > portfolio_value[-2]: 
                profitable_trades += 1
            else:
                loss_trades += 1
        elif weighted_log_return_prediction <= 0 and not holding:
            portfolio_value.append(portfolio_value[-1])

    return np.array(portfolio_value), trades, profitable_trades, loss_trades, trading_days

print("Generating synthetic 1-minute BTCUSDT data...")
filename = r"C:\Users\1jafa\OneDrive\Documents\Quant\Sandbox\first_last_thirty_minutes\data.csv"
df = pd.read_csv(filename, parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
print(f"Generated {len(df)} data points.")

print("Calculating technical indicators...")
df_indicators = calculate_technical_indicators(df.copy())
print("Technical indicators calculated.")

features = [col for col in df_indicators.columns if col not in ['Open', 'High', 'Low', 'Close']]
X = df_indicators[features]

targets = {
    1: df_indicators['Close'].shift(-1),
    2: df_indicators['Close'].shift(-2),
    5: df_indicators['Close'].shift(-5),
    10: df_indicators['Close'].shift(-10),
    20: df_indicators['Close'].shift(-20),
    30: df_indicators['Close'].shift(-30)
}

max_lag = max(targets.keys())
df_final = df_indicators.iloc[:-max_lag].copy()
X_final = X.iloc[:-max_lag].copy()

for n_ahead in targets.keys():
    df_final[f'Target_{n_ahead}'] = targets[n_ahead].iloc[:-max_lag]

train_size = int(len(df_final) * 0.85)
val_size = int(len(df_final) * 0.05)

X_train, y_train_dict = X_final.iloc[:train_size], {n: df_final[f'Target_{n}'].iloc[:train_size] for n in targets.keys()}
X_val, y_val_dict = X_final.iloc[train_size : train_size + val_size], {n: df_final[f'Target_{n}'].iloc[train_size : train_size + val_size] for n in targets.keys()}
X_test, y_test_dict = X_final.iloc[train_size + val_size:], {n: df_final[f'Target_{n}'].iloc[train_size + val_size:] for n in targets.keys()}

xgboost_models = {}
val_predictions_raw = {}

space_xgb = [
    Real(0.01, 0.5, name='learning_rate'),
    Integer(3, 15, name='max_depth'),
    Real(0.0, 1.0, name='gamma'),
    Real(0.0, 1.0, name='subsample'),
    Real(0.0, 1.0, name='colsample_bytree'),
    Real(1e-9, 10.0, name='reg_alpha', prior='log-uniform'),
    Real(1e-9, 10.0, name='reg_lambda', prior='log-uniform')
]

@use_named_args(space_xgb)
def objective_xgb(learning_rate, max_depth, gamma, subsample, colsample_bytree, reg_alpha, reg_lambda):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=learning_rate,
        max_depth=max_depth,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train_current)
    y_pred = model.predict(X_val)
    return mean_squared_error(y_val_current, y_pred)

print("Training XGBoost models and optimizing hyperparameters...")
for n_ahead in targets.keys():
    print(f"Optimizing model for {n_ahead}-minute prediction...")
    y_train_current = y_train_dict[n_ahead]
    y_val_current = y_val_dict[n_ahead]

    res_gp = gp_minimize(objective_xgb, space_xgb, n_calls=20, random_state=42, verbose=False)
    
    best_params = {dim.name: val for dim, val in zip(space_xgb, res_gp.x)}
    
    best_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    best_model.fit(X_train, y_train_current)
    xgboost_models[n_ahead] = best_model
    val_predictions_raw[n_ahead] = pd.Series(best_model.predict(X_val), index=X_val.index)
print("XGBoost models trained and hyperparameters optimized.")

def evaluate_strategy_profit(weights_array):
    weights_dict = {n: w for n, w in zip(targets.keys(), weights_array)}
    
    total_weight = sum(weights_dict.values())
    if total_weight == 0:
        return 1e9 
    normalized_weights = {n: w / total_weight for n, w in weights_dict.items()}

    predictions_for_strategy = {}
    for n_ahead, pred_series in val_predictions_raw.items():
        predictions_for_strategy[n_ahead] = pred_series

    portfolio_values, _, _, _, _ = run_strategy(df_final.loc[X_val.index], predictions_for_strategy, normalized_weights)
    
    if len(portfolio_values) > 1:
        total_profit, _, _ = calculate_metrics(portfolio_values)
        return -total_profit 
    return 1e9

print("Optimizing strategy weights for maximum profit...")
weights_bounds = [(0.0, 1.0)] * len(targets.keys())
res_weights = gp_minimize(evaluate_strategy_profit, weights_bounds, n_calls=30, random_state=42, verbose=False)

optimized_weights_array = res_weights.x
total_weight_opt = sum(optimized_weights_array)
if total_weight_opt == 0:
    optimized_weights = {n: 1/len(targets.keys()) for n in targets.keys()}
else:
    optimized_weights = {n: w / total_weight_opt for n, w in zip(targets.keys(), optimized_weights_array)}

print("Strategy weights optimized.")
print("Optimized Weights:", optimized_weights)

test_predictions_raw = {}
for n_ahead in targets.keys():
    test_predictions_raw[n_ahead] = pd.Series(xgboost_models[n_ahead].predict(X_test), index=X_test.index)

print("Running backtest on the test data...")
algo_portfolio_values, algo_trades, algo_profitable_trades, algo_loss_trades, algo_trading_days = run_strategy(df_final.loc[X_test.index], test_predictions_raw, optimized_weights)

bah_portfolio_values = np.array([1.0] * len(X_test) * 2) 
if len(X_test) > 0:
    bah_portfolio_values = np.array([1.0, df_final.loc[X_test.index]['Close'].iloc[-1] / df_final.loc[X_test.index]['Close'].iloc[0]])
    if len(bah_portfolio_values) == 1:
        bah_portfolio_values = np.array([1.0, 1.0]) 

print("\n--- Strategy Performance ---")
algo_total_profit, algo_sharpe, algo_sortino = calculate_metrics(algo_portfolio_values)
print(f"Algorithmic Strategy Total Profit: {algo_total_profit:.3f}")
print(f"Algorithmic Strategy Sharpe Ratio: {algo_sharpe:.3f}")
print(f"Algorithmic Strategy Sortino Ratio: {algo_sortino:.3f}")
print(f"Algorithmic Strategy Number of Trades: {algo_trades}")
print(f"Algorithmic Strategy Profitable Trades: {algo_profitable_trades}")
print(f"Algorithmic Strategy Loss Trades: {algo_loss_trades}")
print(f"Algorithmic Strategy Trading Days: {algo_trading_days}")

print("\n--- Buy & Hold Performance ---")
bah_total_profit, bah_sharpe, bah_sortino = calculate_metrics(bah_portfolio_values)
print(f"Buy & Hold Total Profit: {bah_total_profit:.3f}")
print(f"Buy & Hold Sharpe Ratio: {bah_sharpe:.3f}")
print(f"Buy & Hold Sortino Ratio: {bah_sortino:.3f}")

if algo_total_profit > bah_total_profit:
    print("\nThe Algorithmic Stratégy *outperformed* the Buy & Hold strategy in terms of Total Profit.")
else:
    print("\nThe Algorithmic Stratégy *did not outperform* the Buy & Hold strategy in terms of Total Profit.")

# IL reste encore pas mal de choses à faire
