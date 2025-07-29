import os
from binance.client import Client
from binance.enums import *
import pandas as pd
from datetime import datetime
import time

class BinanceSimulator:
    def __init__(self, api_key, api_secret, testnet=True):
        """
        Initialize Binance client for simulation
        Always use testnet=True for simulation
        """
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.testnet = testnet
        
    def get_account_info(self):
        """Get account information and balances"""
        try:
            account_info = self.client.get_account()
            print("Account Information:")
            print(f"Can Trade: {account_info['canTrade']}")
            print(f"Can Withdraw: {account_info['canWithdraw']}")
            print(f"Can Deposit: {account_info['canDeposit']}")
            
            print("\nBalances:")
            for balance in account_info['balances']:
                if float(balance['free']) > 0 or float(balance['locked']) > 0:
                    print(f"{balance['asset']}: Free={balance['free']}, Locked={balance['locked']}")
                    
            return account_info
        except Exception as e:
            print(f"Error getting account info: {e}")
            return None
    
    def get_symbol_info(self, symbol):
        """Get trading rules for a symbol"""
        try:
            info = self.client.get_symbol_info(symbol)
            print(f"\nSymbol Info for {symbol}:")
            print(f"Status: {info['status']}")
            
            for filter_info in info['filters']:
                if filter_info['filterType'] == 'LOT_SIZE':
                    print(f"Min Quantity: {filter_info['minQty']}")
                    print(f"Max Quantity: {filter_info['maxQty']}")
                    print(f"Step Size: {filter_info['stepSize']}")
                elif filter_info['filterType'] == 'PRICE_FILTER':
                    print(f"Min Price: {filter_info['minPrice']}")
                    print(f"Max Price: {filter_info['maxPrice']}")
                    print(f"Tick Size: {filter_info['tickSize']}")
                elif filter_info['filterType'] == 'MIN_NOTIONAL':
                    print(f"Min Notional: {filter_info['minNotional']}")
                    
            return info
        except Exception as e:
            print(f"Error getting symbol info: {e}")
            return None
    
    def get_current_price(self, symbol):
        """Get current price for a symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            print(f"Current price of {symbol}: ${price:.2f}")
            return price
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_orderbook(self, symbol, limit=10):
        """Get order book data"""
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            print(f"\nOrder Book for {symbol}:")
            print("Bids (Buy Orders):")
            for bid in depth['bids'][:5]:
                print(f"  Price: {bid[0]}, Quantity: {bid[1]}")
            print("Asks (Sell Orders):")
            for ask in depth['asks'][:5]:
                print(f"  Price: {ask[0]}, Quantity: {ask[1]}")
            return depth
        except Exception as e:
            print(f"Error getting order book: {e}")
            return None
    
    def get_klines(self, symbol, interval='1h', limit=100):
        """Get historical price data (candlesticks)"""
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp to readable date
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                df[col] = df[col].astype(float)
            
            print(f"\nLatest {limit} {interval} candles for {symbol}:")
            print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail())
            
            return df
        except Exception as e:
            print(f"Error getting klines: {e}")
            return None
    
    def simulate_market_buy(self, symbol, quantity):
        try:
            current_price = self.get_current_price(symbol)
            if current_price:
                estimated_cost = current_price * quantity
                print(f"\n=== SIMULATION: Market Buy Order ===")
                print(f"Symbol: {symbol}")
                print(f"Quantity: {quantity}")
                print(f"Estimated Price: ${current_price:.2f}")
                print(f"Estimated Cost: ${estimated_cost:.2f}")
                print("Status: SIMULATED (not executed)")
                
                # ACTUAL ORDER EXECUTION (COMMENTED OUT FOR SIMULATION)
                # order = self.client.order_market_buy(
                #     symbol=symbol,
                #     quantity=quantity
                # )
                # print("Order executed:", order)
                # return order
                
                return {
                    'symbol': symbol,
                    'side': 'BUY',
                    'type': 'MARKET',
                    'quantity': quantity,
                    'price': current_price,
                    'status': 'SIMULATED'
                }
        except Exception as e:
            print(f"Error in simulate_market_buy: {e}")
            return None
    
    def simulate_market_sell(self, symbol, quantity):
        try:
            current_price = self.get_current_price(symbol)
            if current_price:
                estimated_proceeds = current_price * quantity
                print(f"\n=== SIMULATION: Market Sell Order ===")
                print(f"Symbol: {symbol}")
                print(f"Quantity: {quantity}")
                print(f"Estimated Price: ${current_price:.2f}")
                print(f"Estimated Proceeds: ${estimated_proceeds:.2f}")
                print("Status: SIMULATED (not executed)")
                
                # ACTUAL ORDER EXECUTION (COMMENTED OUT FOR SIMULATION)
                # order = self.client.order_market_sell(
                #     symbol=symbol,
                #     quantity=quantity
                # )
                # print("Order executed:", order)
                # return order
                
                return {
                    'symbol': symbol,
                    'side': 'SELL',
                    'type': 'MARKET',
                    'quantity': quantity,
                    'price': current_price,
                    'status': 'SIMULATED'
                }
        except Exception as e:
            print(f"Error in simulate_market_sell: {e}")
            return None
    
    def simulate_limit_buy(self, symbol, quantity, price):
        try:
            current_price = self.get_current_price(symbol)
            total_cost = price * quantity
            
            print(f"\n=== SIMULATION: Limit Buy Order ===")
            print(f"Symbol: {symbol}")
            print(f"Quantity: {quantity}")
            print(f"Limit Price: ${price:.2f}")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Total Cost: ${total_cost:.2f}")
            print("Status: SIMULATED (not executed)")
            
            # ACTUAL ORDER EXECUTION (COMMENTED OUT FOR SIMULATION)
            # order = self.client.order_limit_buy(
            #     symbol=symbol,
            #     quantity=quantity,
            #     price=str(price)
            # )
            # print("Order placed:", order)
            # return order
            
            return {
                'symbol': symbol,
                'side': 'BUY',
                'type': 'LIMIT',
                'quantity': quantity,
                'price': price,
                'status': 'SIMULATED'
            }
        except Exception as e:
            print(f"Error in simulate_limit_buy: {e}")
            return None
    
    def get_open_orders(self, symbol=None):
        """Get open orders"""
        try:
            if symbol:
                orders = self.client.get_open_orders(symbol=symbol)
            else:
                orders = self.client.get_open_orders()
            
            print(f"\nOpen Orders{' for ' + symbol if symbol else ''}:")
            if orders:
                for order in orders:
                    print(f"Order ID: {order['orderId']}")
                    print(f"Symbol: {order['symbol']}")
                    print(f"Side: {order['side']}")
                    print(f"Type: {order['type']}")
                    print(f"Quantity: {order['origQty']}")
                    print(f"Price: {order['price']}")
                    print(f"Status: {order['status']}")
                    print("---")
            else:
                print("No open orders")
                
            return orders
        except Exception as e:
            print(f"Error getting open orders: {e}")
            return None

def main():
    API_KEY, API_SECRET = None, None
    with open('credentials.txt', 'r') as file:
            lines = file.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("API Key:"): API_KEY = line.split("API Key:", 1)[1].strip()
        elif line.startswith("Secret Key:"): API_SECRET = line.split("Secret Key:", 1)[1].strip()
    
    simulator = BinanceSimulator(API_KEY, API_SECRET, testnet=True)
    
    symbol = "BTCUSDT"
    
    print("=== Binance Trading Simulator ===")
    
    simulator.get_account_info()
    
    simulator.get_symbol_info(symbol)
    
    current_price = simulator.get_current_price(symbol)
    
    simulator.get_orderbook(symbol)
    
    df = simulator.get_klines(symbol, interval='1h', limit=10)
    
    simulator.simulate_market_buy(symbol, 0.001)  # Buy 0.001 BTC
    time.sleep(1)
    simulator.simulate_limit_buy(symbol, 0.001, current_price * 0.95)  # Limit buy 5% below current price
    time.sleep(1)
    simulator.simulate_market_sell(symbol, 0.001)  # Sell 0.001 BTC
    
    # Check open orders
    simulator.get_open_orders(symbol)

if __name__ == "__main__":
    main()
