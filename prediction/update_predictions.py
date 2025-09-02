import gc
import os
import re
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from binance.client import Client
import argparse
from bs4 import BeautifulSoup

from model import KronosTokenizer, Kronos, KronosPredictor

from config import Config




def load_model():
    """Loads the Kronos model and tokenizer."""
    print("Loading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained(Config["MODEL_PATH"] + "NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(Config["MODEL_PATH"] + "NeoQuasar/Kronos-base")
    tokenizer.eval()
    model.eval()
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    print("Model loaded successfully.")
    return predictor


def make_prediction(df, predictor):
    """Generates probabilistic forecasts using the Kronos model."""
    last_timestamp = df['timestamps'].max()
    start_new_range = last_timestamp + pd.Timedelta(hours=1)
    new_timestamps_index = pd.date_range(
        start=start_new_range,
        periods=Config["PRED_HORIZON"],
        freq='h'
    )
    y_timestamp = pd.Series(new_timestamps_index, name='y_timestamp')
    x_timestamp = df['timestamps']
    x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]

    with torch.no_grad():
        print("Making main prediction (T=1.0)...")
        begin_time = time.time()
        close_preds_main, volume_preds_main = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=Config["PRED_HORIZON"], T=1.0, top_p=0.95,
            sample_count=Config["N_PREDICTIONS"], verbose=True
        )
        print(f"Main prediction completed in {time.time() - begin_time:.2f} seconds.")

        # print("Making volatility prediction (T=0.9)...")
        # begin_time = time.time()
        # close_preds_volatility, _ = predictor.predict(
        #     df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        #     pred_len=Config["PRED_HORIZON"], T=0.9, top_p=0.9,
        #     sample_count=Config["N_PREDICTIONS"], verbose=True
        # )
        # print(f"Volatility prediction completed in {time.time() - begin_time:.2f} seconds.")
        close_preds_volatility = close_preds_main

    return close_preds_main, volume_preds_main, close_preds_volatility


def fetch_binance_data(symbol, interval):
    """Fetches K-line data from the Binance public API."""
    limit = Config["HIST_POINTS"] + Config["VOL_WINDOW"]

    print(f"Fetching {limit} bars of {symbol} {interval} data from Binance...")
    client = Client()
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(klines, columns=cols)

    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']]
    df.rename(columns={'quote_asset_volume': 'amount', 'open_time': 'timestamps'}, inplace=True)

    df['timestamps'] = pd.to_datetime(df['timestamps'])
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col])

    print("Data fetched successfully.")
    return df


def calculate_metrics(hist_df, close_preds_df, v_close_preds_df):
    """
    Calculates upside and volatility amplification probabilities for the 24h horizon.
    """
    last_close = hist_df['close'].iloc[-1]

    # 1. Upside Probability (for the 24-hour horizon)
    # This is the probability that the price at the end of the horizon is higher than now.
    final_hour_preds = close_preds_df.iloc[-1]
    upside_prob = (final_hour_preds > last_close).mean()

    # 2. Volatility Amplification Probability (over the 24-hour horizon)
    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1))
    historical_vol = hist_log_returns.iloc[-Config["VOL_WINDOW"]:].std()

    amplification_count = 0
    for col in v_close_preds_df.columns:
        full_sequence = pd.concat([pd.Series([last_close]), v_close_preds_df[col]]).reset_index(drop=True)
        pred_log_returns = np.log(full_sequence / full_sequence.shift(1))
        predicted_vol = pred_log_returns.std()
        if predicted_vol > historical_vol:
            amplification_count += 1

    vol_amp_prob = amplification_count / len(v_close_preds_df.columns)

    print(f"Upside Probability (24h): {upside_prob:.2%}, Volatility Amplification Probability: {vol_amp_prob:.2%}")
    return upside_prob, vol_amp_prob


def create_plot(hist_df, close_preds_df, volume_preds_df,symbol,interval):
    """Generates and saves a comprehensive forecast chart."""
    print("Generating comprehensive forecast chart...")
    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    hist_time = hist_df['timestamps']
    last_hist_time = hist_time.iloc[-1]
    pred_time = pd.to_datetime([last_hist_time + timedelta(hours=i + 1) for i in range(len(close_preds_df))])

    ax1.plot(hist_time, hist_df['close'], color='royalblue', label='Historical Price', linewidth=1.5)
    mean_preds = close_preds_df.mean(axis=1)
    ax1.plot(pred_time, mean_preds, color='darkorange', linestyle='-', label='Mean Forecast')
    ax1.fill_between(pred_time, close_preds_df.min(axis=1), close_preds_df.max(axis=1), color='darkorange', alpha=0.2, label='Forecast Range (Min-Max)')
    # 生成一个当前时间的字符串，然后拼接进title
    ax1.set_title(f'{Config["SYMBOL"]} Probabilistic Price & Volume Forecast (Next {interval}) - Build Time:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=16, weight='bold')
    #ax1.set_title(f'{Config["SYMBOL"]} Probabilistic Price & Volume Forecast (Next 15 minute)', fontsize=14, weight='bold')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2.bar(hist_time, hist_df['volume'], color='skyblue', label='Historical Volume', width=0.03)
    ax2.bar(pred_time, volume_preds_df.mean(axis=1), color='sandybrown', label='Mean Forecasted Volume', width=0.03)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time (UTC)')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    separator_time = hist_time.iloc[-1] + timedelta(minutes=30)
    for ax in [ax1, ax2]:
        ax.axvline(x=separator_time, color='red', linestyle='--', linewidth=1.5, label='_nolegend_')
        ax.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    chart_path = Config["REPO_PATH"] / f'img/{symbol}/{interval}' / f'prediction_chart_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.png'
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"Chart saved to: {chart_path}")





def update_html(upside_prob, vol_amp_prob,symbol,interval):
    """
    Updates the btc_index.html file with the latest metrics and timestamp.
    Uses BeautifulSoup to parse and modify HTML.
    """
    print("Updating btc_index.html...")
    html_path = Config["REPO_PATH"] / f'{symbol}_index_{interval}.html'
    now_beijing = datetime.now(timezone.utc) + timedelta(hours=8)
    now_utc_str = now_beijing.strftime('%Y-%m-%d %H:%M:%S')
    upside_prob_str = f'{upside_prob:.1%}'
    vol_amp_prob_str = f'{vol_amp_prob:.1%}'

    # 获取图片列表并按名称倒序排列
    img_dir = Config["REPO_PATH"] / f'img/{symbol}/{interval}'
    img_files = sorted(img_dir.glob(f'prediction_chart_*.png'), reverse=True)

    # 读取HTML文件
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # 更新时间戳
    update_time_element = soup.find('strong', id='update-time')
    if update_time_element:
        update_time_element.string = now_utc_str

    # 更新上涨概率
    upside_prob_element = soup.find('p', class_='metric-value', id='upside-prob')
    if upside_prob_element:
        upside_prob_element.string = upside_prob_str

    # 更新波动率放大概率
    vol_amp_prob_element = soup.find('p', class_='metric-value', id='vol-amp-prob')
    if vol_amp_prob_element:
        vol_amp_prob_element.string = vol_amp_prob_str

    # 更新图片容器部分
    container_list = soup.find('div', class_='container-list')
    if container_list:
        # 清空现有内容
        container_list.clear()

        # 添加新的图表容器
        for img_file in img_files[:300]:  # 限制最多显示10张图片
            chart_div = soup.new_tag('div', **{'class': 'chart-container'})
            img_tag = soup.new_tag('img',
                                   src=f'img/{symbol}/{interval}/{img_file.name}',
                                   **{'class': 'chart-img'})
            chart_div.append(img_tag)
            container_list.append(chart_div)

    # 写回文件
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(str(soup.prettify()))

    print("HTML file updated successfully.")


def main_task(model, symbol, interval):
    """Executes one full update cycle."""
    print("\n" + "=" * 60 + f"\nStarting update task at {datetime.now(timezone.utc)}\n" + "=" * 60)

    # Update Config with command line arguments if provided
    Config["SYMBOL"] = symbol
    Config["INTERVAL"] = interval

    df_full = fetch_binance_data(symbol, interval)
    df_for_model = df_full.iloc[:-1]

    close_preds, volume_preds, v_close_preds = make_prediction(df_for_model, model)

    hist_df_for_plot = df_for_model.tail(Config["HIST_POINTS"])
    hist_df_for_metrics = df_for_model.tail(Config["VOL_WINDOW"])

    upside_prob, vol_amp_prob = calculate_metrics(hist_df_for_metrics, close_preds, v_close_preds)
    create_plot(hist_df_for_plot, close_preds, volume_preds,symbol,interval)
    update_html(upside_prob, vol_amp_prob,symbol,interval)

    commit_message = f"Auto-update forecast for {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
    git_commit_and_push(commit_message,symbol, interval)

    # --- 新增的内存清理步骤 ---
    # 显式删除大的DataFrame对象，帮助垃圾回收器
    del df_full, df_for_model, close_preds, volume_preds, v_close_preds
    del hist_df_for_plot, hist_df_for_metrics

    # 强制执行垃圾回收
    gc.collect()
    # --- 内存清理结束 ---

    print("-" * 60 + "\n--- Task completed successfully ---\n" + "-" * 60 + "\n")


def run_scheduler(model, symbol, interval):
    """A continuous scheduler that runs the main task hourly."""
    while True:
        now = datetime.now()
        next_run_time = (now + timedelta(minutes=1))

        print(f"Current time: {now:%Y-%m-%d %H:%M:%S}.")
        print(f"Next run at: {next_run_time:%Y-%m-%d %H:%M:%S}. Waiting for {60:.0f} seconds...")

        time.sleep(60)

        try:
            main_task(model, symbol, interval)
        except Exception as e:
            print(f"\n!!!!!! A critical error occurred in the main task !!!!!!!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("Retrying in 5 minutes...")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            time.sleep(60)



def git_commit_and_push(commit_message,symbol, interval):
    """Adds, commits, and pushes specified files to the Git repository. BTCUSDT"""
    print("Performing Git operations...")
    try:
        os.chdir(Config["REPO_PATH"])
        #git pull
        subprocess.run(['git', 'pull'], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'add', f'{symbol}_index_{ interval}.html'], check=True, capture_output=True, text=True)
        # 添加 img/btc 目录下的所有文件
        subprocess.run(['git', 'add', f'img/{symbol}/{interval}'], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'commit', '-m', commit_message], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'push'], check=True, capture_output=True, text=True)
        print("Git push successful.")
    except subprocess.CalledProcessError as e:
        output = e.stdout if e.stdout else e.stderr
        if "nothing to commit" in output or "Your branch is up to date" in output:
            print("No new changes to commit or push.")
        else:
            print(f"A Git error occurred:\n--- STDOUT ---\n{e.stdout}\n--- STDERR ---\n{e.stderr}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Update predictions with custom symbol and interval')
    parser.add_argument('--symbol', type=str ,default='BTCUSDT', help='Trading symbol (e.g., BTCUSDT, ETHUSDT)')
    parser.add_argument('--interval', type=str ,default='15m', help='Kline interval (e.g., 15m, 1h, 4h)')
    return parser.parse_args()



if __name__ == '__main__':
    #model_path = Path(Config["MODEL_PATH"])
    #model_path.mkdir(parents=True, exist_ok=True)

    #git_commit_and_push('test-commit')
    args = parse_args()
    # Use command line arguments if provided, otherwise use defaults
    symbol = args.symbol
    interval = args.interval

    if interval == '1h':
        Config['HIST_POINTS'] = 360
        Config['PRED_HORIZON'] = 24
        Config['N_PREDICTIONS'] = 30
        Config['VOL_WINDOW'] = 24

    loaded_model = load_model()
    main_task(loaded_model, symbol, interval)  # Run once on startu
    run_scheduler(loaded_model,symbol, interval)  # Start the schedule