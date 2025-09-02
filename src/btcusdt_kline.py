import asyncio
import os
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
import pymysql
import pandas as pd
import argparse
from db_config import DB_CONFIG


from binance_sdk_spot.spot import (
    Spot,
    SPOT_WS_STREAMS_PROD_URL,
    ConfigurationWebSocketStreams,
)
from binance_sdk_spot.websocket_streams.models import KlineIntervalEnum

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create configuration for the WebSocket Streams
configuration_ws_streams = ConfigurationWebSocketStreams(
    stream_url=os.getenv("STREAM_URL", SPOT_WS_STREAMS_PROD_URL)
)

# Initialize Spot client
client = Spot(config_ws_streams=configuration_ws_streams)

def convert_kline_to_fund_format(kline_data):
    """
    将Binance K线数据转换为fund_daily格式

    Parameters:
    kline_data (KlineResponse): Binance K线数据对象

    Returns:
    dict: 转换后的数据字典
    """
    # 使用对象属性访问而不是字典键访问
    k = kline_data.k

    # 计算变化值和变化百分比
    open_price = float(k.o)
    close_price = float(k.c)
    change = close_price - open_price
    pct_chg = (change / open_price) * 100 if open_price != 0 else 0

    # 格式化交易日期 (使用K线开始时间)
    #trade_date = datetime.fromtimestamp(k.T / 1000).strftime('%Y%m%d')
    trade_date = datetime.fromtimestamp((k.T + 1) / 1000).strftime('%Y-%m-%d %H:%M:%S')

    # 映射字段
    converted_data = {
        "ts_code": kline_data.s ,  # 添加T标识币种
        "trade_date": trade_date,
        "pre_close": open_price,  # 使用开盘价作为前收盘价
        "open": open_price,
        "high": float(k.h),
        "low": float(k.l),
        "close": close_price,
        "change": change,
        "pct_chg": pct_chg,
        "vol": float(k.v),  # 成交量
        "amount": float(k.q)  # 成交额
    }

    return converted_data


def handle_kline_message(data,interval=None):
    """
    处理K线数据消息的方法
    """
    print(f"转换后的数据a: {data}")

    try:
        # 检查事件类型
        if hasattr(data, 'e') and data.e == 'kline':
            # 转换数据格式
            converted_data = convert_kline_to_fund_format(data)
            print(f"转换后的数据: {converted_data}")

            # 数据库配置信息

            # 保存到MySQL
            save_kline_to_mysql(converted_data, DB_CONFIG,interval)

    except Exception as e:
        logging.error(f"数据处理错误: {e}")

async def kline(symbol, interval):
    connection = None
    try:
        connection = await client.websocket_streams.create_connection()

        stream = await connection.kline(
            symbol=symbol,
            interval=interval,
        )
        stream.on("message", lambda data: handle_kline_message(data, interval))
        await asyncio.sleep(3)
        # 使用无限循环保持连接持续运行
        #while True:
        #    await asyncio.sleep(1)  # 保持连接活跃

    except Exception as e:
        logging.error(f"kline() error: {e}")
    finally:
        if connection:
            await connection.close_connection(close_session=True)


def save_kline_to_mysql(data, db_config, interval=None):
    """
    保存单条K线数据到MySQL

    Parameters:
    data (dict): 转换后的K线数据
    db_config (dict): 数据库连接配置
    """

    # 创建数据库连接引擎
    engine = create_engine(
        f"mysql+pymysql://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )

    # 添加时间戳字段
    data['gmt_create'] = datetime.now()
    data['gmt_update'] = datetime.now()
    # 使用传入的interval参数
    data['iinterval'] = interval if interval else '15m'

    # 转换为DataFrame
    df = pd.DataFrame([data])

    # 对MySQL保留关键字添加反引号
    columns = list(df.columns)
    escaped_columns = [f'`{col}`' if col in ['change'] else col for col in columns]
    column_str = ', '.join(escaped_columns)
    placeholders = ', '.join([f':{col}' for col in columns])

    # 构造REPLACE INTO语句
    sql = f"REPLACE INTO new_kline_data ({column_str}) VALUES ({placeholders})"

    # 执行插入
    with engine.connect() as conn:
        conn.execute(text(sql), df.to_dict('records')[0])
        conn.commit()
    print(f"K线数据已保存到MySQL")


def get_interval_enum(interval_str):
    """
    将字符串转换为KlineIntervalEnum枚举值

    Parameters:
    interval_str (str): 时间间隔字符串，如 "15m", "1h" 等

    Returns:
    str: 对应的枚举值
    """
    # 构造枚举键名
    enum_key = f"INTERVAL_{interval_str.upper()}"

    # 特殊处理带数字的间隔，如15m, 1h等
    if 'M' in interval_str.upper():
        enum_key = f"INTERVAL_{interval_str.upper().replace('M', 'm')}"
    elif 'H' in interval_str.upper():
        enum_key = f"INTERVAL_{interval_str.upper().replace('H', 'h')}"

    try:
        return KlineIntervalEnum[enum_key].value
    except KeyError:
        # 如果找不到对应的枚举值，返回默认值
        return KlineIntervalEnum["INTERVAL_15m"].value



if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='获取Binance K线数据并保存到MySQL')
    parser.add_argument('--symbol', type=str, default='bnbusdt', help='交易对符号，默认: bnbusdt')
    parser.add_argument('--interval', type=str, default='15m', help='K线时间间隔，默认: 15m')

    # 解析命令行参数
    args = parser.parse_args()

    # 获取参数值
    symbol = args.symbol
    interval = get_interval_enum(args.interval)

    print(f"开始获取 {symbol} 的 {args.interval} K线数据...")

    # 运行异步函数
    asyncio.run(kline(symbol, interval))
