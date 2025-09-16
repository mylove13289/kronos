import os
import pickle
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
from tqdm import trange, tqdm
import time
import random

from config import Config


class BTCDataPreprocessor:
    """
    A class to handle the loading, processing, and splitting of BTC financial data from Binance.
    """

    def __init__(self):
        """Initializes the preprocessor with configuration."""
        self.config = Config()
        self.data = {}  # A dictionary to store processed data for each symbol.
        # A list of official Binance API endpoints for redundancy.
        self.api_endpoints = [
                                 "https://api.binance.com",
                                 # "https://api1.binance.com",
                                 # "https://api2.binance.com",
                                 # "https://api3.binance.com",
                             ] * 100000

    def fetch_binance_data(self, symbol='BTCUSDT', interval='1h', start_time=None, end_time=None):
        """
        从Binance API获取历史K线数据

        Args:
            symbol (str): 交易对符号
            interval (str): K线间隔 ('1h', '1d', etc.)
            start_time (str): 开始时间 'YYYY-MM-DD'
            end_time (str): 结束时间 'YYYY-MM-DD'

        Returns:
            pd.DataFrame: OHLCV数据
        """
        print(f"Fetching {symbol} data from Binance...")

        # 转换时间格式
        if start_time:
            start_ts = int(pd.Timestamp(start_time).timestamp() * 1000)
        else:
            start_ts = None

        if end_time:
            end_ts = int(pd.Timestamp(end_time).timestamp() * 1000)
        else:
            end_ts = None

        all_data = []
        current_start = start_ts
        limit = 1000  # Binance API限制每次最多1000条

        while True:
            # 构建请求参数
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            if current_start:
                params['startTime'] = current_start
            if end_ts:
                params['endTime'] = end_ts

            data = None
            for base_url in self.api_endpoints:
                endpoint_url = f"{base_url}/api/v3/klines"
                try:
                    # 发送请求，增加10秒超时
                    print(f"Attempting to fetch data from {endpoint_url}...")
                    response = requests.get(endpoint_url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    print(f"Successfully fetched a chunk from {endpoint_url}.")
                    # 请求成功，跳出内部的for循环
                    break
                except requests.exceptions.RequestException as e:
                    print(f"Failed to fetch from {endpoint_url}: {e}")
                    if base_url != self.api_endpoints[-1]:
                        print("Trying next endpoint...")

                    time.sleep(random.randint(1, 3))  # 等待1秒再试下一个节点

            # 如果尝试完所有节点，data仍然是None，说明本次请求失败，无法继续
            if data is None:
                print("All API endpoints failed for the current request. Aborting.")
                break  # 中断获取数据的while循环

            if not data:
                break

            all_data.extend(data)
            print(f"Fetched {len(data)} records, total: {len(all_data)}")

            # 更新起始时间为最后一条记录的时间+1
            if len(data) < limit:
                break

            current_start = data[-1][0] + 1  # 最后一条记录的时间戳+1毫秒

            # 如果已经超过结束时间，则停止
            if end_ts and current_start >= end_ts:
                break

            # 避免请求过于频繁
            time.sleep(0.1)

        if not all_data:
            raise ValueError("No data retrieved from Binance API")

        # 转换为DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # 选择需要的列并转换数据类型
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 转换为数值类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 去除重复和缺失值
        df = df.drop_duplicates(subset=['timestamp']).dropna()
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"Successfully fetched {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df

    def load_btc_data(self, interval='1h'):
        """
        Loads BTC data from Binance API and processes it.
        """
        print("Loading and processing BTC data from Binance...")

        # 获取BTC数据
        try:
            symbol_df = self.fetch_binance_data(
                symbol=self.config.symbol,
                interval=interval,
                start_time=self.config.dataset_begin_time,
                end_time=self.config.dataset_end_time
            )
        except Exception as e:
            print(f"Error fetching BTC data: {e}")
            print("Trying to load from local backup if available...")
            backup_file = 'btc_backup_data.csv'
            if os.path.exists(backup_file):
                symbol_df = pd.read_csv(backup_file)
                symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp'])
                print(f"Loaded backup data with {len(symbol_df)} records")
            else:
                raise ValueError("Could not fetch BTC data and no backup available")

        # 设置datetime为索引
        symbol_df = symbol_df.set_index('timestamp')

        # 重命名列以匹配原始框架格式
        column_mapping = {
            'volume': 'vol'
        }
        symbol_df = symbol_df.rename(columns=column_mapping)

        # 添加amount特征（成交额 = OHLC平均价 × 成交量，与qlib方法保持一致）
        symbol_df['amt'] = ((symbol_df['open'] + symbol_df['high'] +
                                symbol_df['low'] + symbol_df['close']) / 4) * symbol_df['vol']
        print(f"Added amount feature (OHLC average * volume) to match qlib data")

        # 添加时间特征（基于时间戳索引）
        print("Adding time features...")
        symbol_df['minute'] = symbol_df.index.minute
        symbol_df['hour'] = symbol_df.index.hour
        symbol_df['weekday'] = symbol_df.index.weekday  # 0=Monday, 6=Sunday
        symbol_df['day'] = symbol_df.index.day
        symbol_df['month'] = symbol_df.index.month
        symbol_df['year'] = symbol_df.index.year
        print(f"Added time features: {self.config.time_feature_list}")

        # 选择最终特征（价格特征 + 时间特征）
        final_features = self.config.feature_list + self.config.time_feature_list
        symbol_df = symbol_df[final_features]

        # 过滤掉数据不足的部分
        symbol_df = symbol_df.dropna()

        # 保存备份
        backup_file = 'data/btc_backup_data.csv'
        os.makedirs(os.path.dirname(backup_file), exist_ok=True)  # 添加这一行
        symbol_df.reset_index().to_csv(backup_file, index=False)
        print(f"Saved backup data to {backup_file}")

        # 存储处理后的数据
        self.data[self.config.symbol] = symbol_df
        print(f"Processed BTC data: {len(symbol_df)} records from {symbol_df.index.min()} to {symbol_df.index.max()}")

    def prepare_dataset(self):
        """
        Splits the loaded data into train, validation, and test sets and saves them to disk.
        """
        print("Splitting BTC data into train, validation, and test sets...")
        train_data, val_data, test_data = {}, {}, {}

        symbol = self.config.symbol
        symbol_df = self.data[symbol]

        # Define time ranges from config.
        train_start, train_end = self.config.train_time_range
        val_start, val_end = self.config.val_time_range
        test_start, test_end = self.config.test_time_range

        # Create boolean masks for each dataset split.
        train_mask = (symbol_df.index >= train_start) & (symbol_df.index <= train_end)
        val_mask = (symbol_df.index >= val_start) & (symbol_df.index <= val_end)
        test_mask = (symbol_df.index >= test_start) & (symbol_df.index <= test_end)

        # Apply masks to create the final datasets.
        train_data[symbol] = symbol_df[train_mask]
        val_data[symbol] = symbol_df[val_mask]
        test_data[symbol] = symbol_df[test_mask]

        # 打印数据集信息
        print(f"Dataset split summary:")
        print(f"  Train: {len(train_data[symbol])} records ({train_start} to {train_end})")
        print(f"  Val:   {len(val_data[symbol])} records ({val_start} to {val_end})")
        print(f"  Test:  {len(test_data[symbol])} records ({test_start} to {test_end})")

        # Save the datasets using pickle.
        os.makedirs(self.config.dataset_path, exist_ok=True)

        with open(f"{self.config.dataset_path}/train_data.pkl", 'wb') as f:
            pickle.dump(train_data, f)
        with open(f"{self.config.dataset_path}/val_data.pkl", 'wb') as f:
            pickle.dump(val_data, f)
        with open(f"{self.config.dataset_path}/test_data.pkl", 'wb') as f:
            pickle.dump(test_data, f)

        print("BTC datasets prepared and saved successfully.")
        print(f"Files saved in: {self.config.dataset_path}")

    def verify_data(self):
        """验证生成的数据集"""
        print("\nVerifying generated datasets...")

        for split in ['train', 'val', 'test']:
            file_path = f"{self.config.dataset_path}/{split}_data.pkl"
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                symbol = self.config.symbol
                if symbol in data:
                    df = data[symbol]
                    print(f"{split.upper()} dataset:")
                    print(f"  Shape: {df.shape}")
                    print(f"  Time range: {df.index.min()} to {df.index.max()}")
                    print(f"  Features: {list(df.columns)}")
                    print(f"  Sample data:")
                    print(f"    {df.head(2)}")
                    print()
            else:
                print(f"Warning: {file_path} not found")


if __name__ == '__main__':
    # This block allows the script to be run directly to perform data preprocessing.
    print("🚀 Starting BTC data preprocessing...")

    preprocessor = BTCDataPreprocessor()

    # 打印配置信息
    #preprocessor.config.print_config_summary()

    try:
        # 加载和处理BTC数据
        preprocessor.load_btc_data(interval='15m')

        # 准备训练数据集
        preprocessor.prepare_dataset()

        # 验证数据
        preprocessor.verify_data()

        print("✅ BTC data preprocessing completed successfully!")

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        raise
