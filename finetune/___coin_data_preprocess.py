import os
import pickle
import pandas as pd
from tqdm import trange
from sqlalchemy import create_engine, text

from config import Config

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'db-mysql.dian-stable.com'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'longquan'),
    'password': os.getenv('DB_PASSWORD', 'longquan'),
    'database': os.getenv('DB_NAME', 'test')
}

class QlibDataPreprocessor:
    """
    A class to handle the loading, processing, and splitting of Qlib financial data.
    """

    def __init__(self):
        """Initializes the preprocessor with configuration and data fields."""
        self.config = Config()
        self.data_fields = ['open', 'close', 'high', 'low', 'volume', 'vwap']
        self.data = {}  # A dictionary to store processed data for each symbol.



    def load_qlib_data(self):

        engine = create_engine(
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )

        symbolList = ["ETHUSDT"]
        #symbolList = ["AVAXUSDT","BCHUSDT","BNBUSDT","BTCUSDT","DOGEUSDT","ETHUSDT","LINKUSDT","LTCUSDT","OPUSDT","SOLUSDT","UNIUSDT","XRPUSDT"]
        #循环symbolList
        for symbol in symbolList:
            print(f'start_fetch_data,{ symbol}')

            # 构建SQL查询语句
            query = "SELECT * FROM new_kline_data WHERE 1=1"
            params = {}

            if 1:
                query += " AND ts_code = :symbol"
                params['symbol'] = f'{symbol}'

            if 1:
                query += " AND iinterval = :iinterval"
                params['iinterval'] = '5m'

            query += " ORDER BY trade_date asc"

            # 执行查询并返回DataFrame
            df = pd.read_sql_query(text(query), engine, params=params)
            #删除df的id
            df.drop(columns=['id',"ts_code","iinterval","deleted","gmt_create","gmt_update","change","pct_chg","amount","pre_close"], inplace=True)
            #df.rename(columns={'amount': 'amt'}, inplace=True)
            df['amt'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4 * df['vol']
            #把df中trade_date类型设置为datetime
            df['datetime'] = pd.to_datetime(df['trade_date'])
            #df的ts_code 设置为index
            df.set_index('datetime', inplace=True)
            self.data[symbol] = df
            print(f'end_fetch_data,{ symbol}')


    def prepare_dataset(self):
        """
        Splits the loaded data into train, validation, and test sets and saves them to disk.
        """
        print("Splitting data into train, validation, and test sets...")
        train_data, val_data, test_data = {}, {}, {}

        symbol_list = list(self.data.keys())
        for i in trange(len(symbol_list), desc="Preparing Datasets"):
            symbol = symbol_list[i]
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
            print(f'prepare_dataset,{ symbol}')


        # Save the datasets using pickle.
        os.makedirs(self.config.dataset_path, exist_ok=True)
        with open(f"{self.config.dataset_path}/train_data.pkl", 'wb') as f:
            pickle.dump(train_data, f)
        with open(f"{self.config.dataset_path}/val_data.pkl", 'wb') as f:
            pickle.dump(val_data, f)
        with open(f"{self.config.dataset_path}/test_data.pkl", 'wb') as f:
            pickle.dump(test_data, f)

        print("Datasets prepared and saved successfully.")


if __name__ == '__main__':
    # This block allows the script to be run directly to perform data preprocessing.
    preprocessor = QlibDataPreprocessor()
    #preprocessor.initialize_qlib()
    preprocessor.load_qlib_data()
    preprocessor.prepare_dataset()
