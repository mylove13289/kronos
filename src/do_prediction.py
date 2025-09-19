# do_prediction.py
import os
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

from db_config import DB_CONFIG

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 或者添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib.dates as mdates

from model import Kronos, KronosTokenizer, KronosPredictor
import numpy as np
import torch




def plot_prediction(kline_df, pred_df, symbol, titleFlag, save_path=None):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 绘制价格图表
    ax1.plot(close_df.index, close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df.index, close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)
    ax1.set_title(f'{symbol} Price Prediction ({titleFlag})', fontsize=16)

    # 绘制成交量图表
    ax2.plot(volume_df.index, volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df.index, volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)
    ax2.set_xlabel('Time', fontsize=14)

    # 设置x轴时间格式
    # 根据数据的时间间隔选择合适的格式
    if len(kline_df) > 0:
        date_format = mdates.DateFormatter('%m-%d %H:%M')
        ax2.xaxis.set_major_formatter(date_format)
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())

        # 旋转日期标签以避免重叠
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果图片已保存到: {save_path}")

    plt.show()


def load_kline_data_from_db(db_config, symbol=None, iinterval=None, start_date=None, end_date=None,count=0):
    """
    从数据库查询K线数据

    Parameters:
    db_config (dict): 数据库连接配置
    symbol (str): 交易对符号，如 'BTCUSDT'
    start_date (str): 开始日期，格式 'YYYY-MM-DD'
    end_date (str): 结束日期，格式 'YYYY-MM-DD'

    Returns:
    pandas.DataFrame: 查询到的数据
    """
    # 创建数据库连接引擎
    engine = create_engine(
        f"mysql+pymysql://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )

    # 构建SQL查询语句

    # 执行查询并返回DataFrame
    #df = pd.read_sql_query(text(query), engine, params=params)
    df = pd.read_sql_query(f'select * from (SELECT * FROM new_kline_data WHERE 1=1 AND ts_code = \'{symbol}\' AND iinterval = \'{iinterval}\' and trade_date <= \'{end_date}\' order by id desc limit {count})t order by id asc ', engine)
    # 576 + 72
    # 将查出来的数据按trade_date进行升序
    #df['trade_date'] = pd.to_datetime(df['trade_date'])
    #df.sort_values(by='trade_date', inplace=True)
    return df


# 使用示例
# df = load_kline_data_from_db(DB_CONFIG, symbol='BTCUSDT', start_date='2023-01-01', end_date='2023-12-31')

import argparse


def main(symbol, iinterval , lookback , pred_len,endDate ):
    """
    为什么是42？
    42这个数字在计算机科学和数学领域中具有特殊的文化意义：
    《银河系漫游指南》：在道格拉斯·亚当斯的科幻小说《银河系漫游指南》中，42是"生命、宇宙以及一切"的终极答案
    程序员文化：这个数字在程序员社区中被广泛使用，成为了一种传统和梗
    任意性：实际上任何固定的整数都可以作为随机种子
    :return:
    """
    # 设置随机种子以确保结果可复现
    """     """
    np.random.seed(42)
    torch.manual_seed(42)

    # 如果使用GPU，也设置相应的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Load Model and Tokenizer
    tokenizer = KronosTokenizer.from_pretrained("/Users/longquan/Documents/MYSELF/models/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("/Users/longquan/Documents/MYSELF/models/Kronos-base")

    #tokenizer = KronosTokenizer.from_pretrained("/Users/longquan/Downloads/pipeline_run_20250914_052908_/pipeline_run_20250914_052908_tokenizer/checkpoints/best_model")
    #model = Kronos.from_pretrained("/Users/longquan/Downloads/pipeline_run_20250914_052908_/pipeline_run_20250914_052908_predictor/checkpoints/epoch_0019")


    # 2. Instantiate Predictor
    predictor = KronosPredictor(model, tokenizer, device="mps:0", max_context=512)

    count = lookback + pred_len
    # 3. Prepare Data
    df = load_kline_data_from_db(DB_CONFIG, symbol=symbol, iinterval=iinterval, start_date=None, end_date=endDate,count=count)

    # 添加数据验证
    if df.empty:
        print(f"错误: 没有找到 symbol={symbol}, iinterval={iinterval} 的数据")
        return

    if len(df) < lookback + pred_len:
        print(f"错误: 数据不足。需要 {lookback + pred_len} 行数据，但只找到 {len(df)} 行")
        return

    print(f"成功加载 {len(df)} 行数据")

    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['volume'] = df['vol']

    x_df = df.loc[:lookback - 1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.loc[:lookback - 1, 'trade_date']
    y_timestamp = df.loc[lookback:lookback + pred_len - 1, 'trade_date']

    # 创新性预测（更多样化，适合探索性分析）
    # 平衡预测（推荐的默认设置）
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=0.75,
        top_p=0.92,
        sample_count=8,
        verbose=True
    )

    # 5. Visualize Results
    # Combine historical and forecasted data for plotting
    kline_df = df.loc[:lookback + pred_len - 1]

    # 创建保存路径
    os.makedirs("predictions", exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"predictions/{symbol}_{iinterval}_prediction_{timestamp}.png"

    titleFlag = f'{iinterval} -- {lookback}/{pred_len} -- {endDate}'
    # visualize
    plot_prediction(kline_df, pred_df, symbol,titleFlag, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kronos Model Prediction')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument('--iinterval', type=str, default='15m', help='Interval (e.g., 5m, 15m)')
    parser.add_argument('--lookback', type=int, default=180, help='Lookback period')
    parser.add_argument('--pred_len', type=int, default=45, help='Prediction length')
    parser.add_argument('--playback', type=int, default=15, help='回测')


    args = parser.parse_args()

    # 将 datetime.now() 改为固定时间，例如 '2025-09-15 00:00:00'
    fixed_time = datetime.strptime('2025-09-15 00:00:00', '%Y-%m-%d %H:%M:%S')
    endDate = fixed_time - timedelta(minutes=args.playback * 15)
    endDate = endDate.strftime('%Y-%m-%d %H:%M:%S')

    print(f"查询参数为：symbol={args.symbol}, iinterval={args.iinterval}, lookback={args.lookback}, pred_len={args.pred_len}")

    main(symbol=args.symbol, iinterval=args.iinterval, lookback=args.lookback, pred_len=args.pred_len,endDate=endDate)
