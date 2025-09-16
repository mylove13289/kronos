import asyncio
import os
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import pymysql
import pandas as pd
from db_config import DB_CONFIG
from binance.client import Client
from db_config import DB_CONFIG
import sys

def kline(symbol,iinterval,startDate,endDate):
    """
    def get_historical_klines(
        self,
        symbol,
        interval,
        start_str=None,
        end_str=None,
        limit=None,
        klines_type: HistoricalKlinesType = HistoricalKlinesType.SPOT,
    )
    :param symbol:
    :param interval:
    :return:
    """
    client = Client()
    klines = client.get_historical_klines(symbol, iinterval, startDate, endDate)

    # 如果klines是一个空数组，则返回
    if not klines:
        print(f"No data fetched for {symbol} from {startDate} to {endDate}")
        return

    #klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    # newline = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE, "2020-01-01", "2025-09-01")

    #        :return: list of OHLCV values (Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base asset volume, Taker buy quote asset volume, Ignore)

    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(klines, columns=cols)

    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']]
    df.rename(columns={'quote_asset_volume': 'amount', 'open_time': 'timestamps'}, inplace=True)

    df['trade_date'] = pd.to_datetime(df['timestamps'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')

    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col])

    df['gmt_create'] = datetime.now()
    df['gmt_update'] = datetime.now()
    # 使用传入的interval参数
    df['iinterval'] = iinterval
    df['ts_code'] = symbol
    #DOGEUSDT，XRPUSDT,LINKUSDT,OPUSDT,AVAXUSDT,BCHUSDT,UNIUSDT,LTCUSDT,UNIUSDT

    # 删除timestamps列
    df.drop(columns=['timestamps'], inplace=True)
    #把volume改名为vol
    df.rename(columns={'volume': 'vol'}, inplace=True)

    print("Data fetched successfully.")

    # 创建数据库连接引擎
    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

    columns = list(df.columns)
    escaped_columns = [f'`{col}`' if col in ['change'] else col for col in columns]
    column_str = ', '.join(escaped_columns)
    placeholders = ', '.join([f':{col}' for col in columns])

    # 构造REPLACE INTO语句
    sql = f"REPLACE INTO new_kline_data ({column_str}) VALUES ({placeholders})"

    # 执行插入
    with engine.connect() as conn:
        conn.execute(text(sql), df.to_dict('records'))
        conn.commit()
    print(f"K线数据已保存到MySQL")


def kline_loop(symbol,iinterval,start_date, end_date):
    """
    循环调用kline函数，每次处理一天的数据
    :param start_date: 开始日期，格式 'YYYY-MM-DD'
    :param end_date: 结束日期，格式 'YYYY-MM-DD'
    """
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    while current_date <= end_date_obj+timedelta(days=1):
        # 计算下一天
        next_date = current_date + timedelta(days=1)

        # 格式化日期字符串
        current_date_str = current_date.strftime('%Y-%m-%d')
        next_date_str = next_date.strftime('%Y-%m-%d')

        print(f"正在处理{symbol} :  {current_date_str} 的数据...")

        # 调用kline函数处理当天数据
        kline(symbol,iinterval,current_date_str, next_date_str)

        # 移动到下一天
        current_date = next_date


if __name__ == "__main__":
    # 开始时间和结束时间
    start_date = '2025-09-05'
    end_date = '2025-09-15'
    symbol = 'BTCUSDT'
    iinterval = '15m'
    #symbol = sys.argv[1]
    #iinterval = sys.argv[2]
    # BTCUSDT,ETHUSDT , BNBUSDT,SOLUSDT，LTCUSDT, DOGEUSDT，XRPUSDT,LINKUSDT,OPUSDT,AVAXUSDT,BCHUSDT,UNIUSDT,

    # 循环调用kline函数
    kline_loop(symbol,iinterval,start_date, end_date)