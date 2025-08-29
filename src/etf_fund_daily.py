# 导入tushare
import tushare as ts

from sqlalchemy import create_engine, text
import pymysql
from datetime import datetime
import pandas as pd

# 初始化pro接口
pro = ts.pro_api('6ec6092a6816b497cb6c54214c5c26e12a83a245b9169447e0fea08f')

def get_fund_daily(ts_code):
    """
    获取ETF基金每日行情数据

    Parameters:
    ts_code (str): 基金代码，例如 "510500.SH"

    Returns:
    DataFrame: 基金每日行情数据
    """
    df = pro.fund_daily(**{
        "trade_date": "",
        "start_date": 20250101,
        "end_date": 20250826,
        "ts_code": ts_code,
        "limit": 1000,
        "offset": ""
    }, fields=[
        "ts_code",
        "trade_date",
        "pre_close",
        "open",
        "high",
        "low",
        "close",
        "change",
        "pct_chg",
        "vol",
        "amount"
    ])
    return df

def save_to_mysql_replace(df, table_name, db_config):
    """
    使用真正的REPLACE INTO语句保存数据到MySQL

    Parameters:
    df (DataFrame): 要保存的数据
    table_name (str): 数据库表名
    db_config (dict): 数据库连接配置

    Returns:
    None
    """
    # 创建数据库连接引擎
    engine = create_engine(
        f"mysql+pymysql://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )

    # 获取列名，并对保留关键字添加反引号
    columns = list(df.columns)
    # 对MySQL保留关键字添加反引号
    escaped_columns = [f'`{col}`' if col in ['change'] else col for col in columns]
    column_str = ', '.join(escaped_columns)
    placeholders = ', '.join([f':{col}' for col in columns])

    # 构造REPLACE INTO语句
    sql = f"REPLACE INTO {table_name} ({column_str}) VALUES ({placeholders})"

    # 将DataFrame转换为字典记录列表
    data = df.to_dict('records')

    # 执行批量插入
    with engine.connect() as conn:
        # 使用executemany处理多条记录
        conn.execute(text(sql), data)
        conn.commit()
    print(f"数据已使用REPLACE INTO保存到MySQL数据库表 {table_name} 中")


# 使用示例
if __name__ == "__main__":
    # 调用方法获取指定ts_code的数据
    df = get_fund_daily("510500.SH")

    df['gmt_create'] = datetime.now()
    df['gmt_update'] = datetime.now()
    df['iinterval'] = '1d'

    # 数据库配置信息
    db_config = {
        'host': '8.216.81.73',
        'port': 3306,
        'user': 'root',
        'password': 'Ff123456fx',
        'database': 'zero'
    }

    # 保存数据到数据库
    save_to_mysql_replace(df, 'new_kline_data', db_config)

    print(df)


