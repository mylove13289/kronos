import os

class BTCConfig:
    """
    Configuration class for BTC fine-tuning.
    """

    def __init__(self):
        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        # BTC数据参数
        self.symbol = 'BTCUSDT'
        self.data_source = 'binance'  # 使用Binance数据源
        
        # 时间范围设置
        self.dataset_begin_time = "2025-08-01"
        self.dataset_end_time = '2025-09-10'
        
        # 滑动窗口参数
        self.lookback_window = 360  # 15天历史数据(1h间隔: 15*24=360)
        self.predict_window = 24   # 24小时预测窗口
        self.max_context = 512     # 最大上下文长度
        
        # 特征列表 (OHLCV + Amount)
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        # 时间特征（添加年份以提高时间精度）
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month', 'year']
        
        # =================================================================
        # Dataset Splitting & Paths
        # =================================================================
        # 训练/验证/测试时间范围2
        self.train_time_range = ["2025-08-01", "2025-09-10"]  # 4年训练数据
        self.val_time_range = ["2025-02-01", "2025-08-01"]    # 验证集有重叠
        self.test_time_range = ["2025-08-01", "2025-09-10"]   # 测试集
        self.backtest_time_range = ["2025-08-01", "2025-09-10"] # 回测期间
        
        # 数据集保存路径
        self.dataset_path = "/Users/longquan/Documents/git_repository/myself/kronos/data/processed_datasets"
        
        # =================================================================
        # Training Hyperparameters  
        # =================================================================
        self.clip = 5.0  # 数据归一化裁剪值
        
        self.epochs = 20  # BTC数据波动性高，减少epoch防止过拟合
        self.log_interval = 10  # 更频繁的日志记录
        self.batch_size = 64   # 适中的batch size
        
        # 每个epoch的样本数
        self.n_train_iter = 20000 * self.batch_size
        self.n_val_iter = 80 * self.batch_size
        
        # 学习率 (BTC数据需要更小的学习率)
        self.tokenizer_learning_rate = 4e-4
        # self.predictor_learning_rate = 2e-5
        self.predictor_learning_rate = 8e-4
        
        # 梯度累积
        self.accumulation_steps = 2  # 增加有效batch size
        
        # AdamW参数
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1
        
        # 随机种子
        self.seed = 42
        
        # =================================================================
        # Experiment Logging & Saving
        # =================================================================
        self.use_comet = False  # 可以设置为True如果你有Comet ML账号
        self.comet_config = {
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-BTC-Finetune",
            "workspace": "your_workspace"
        }
        self.comet_tag = 'btc_finetune'
        self.comet_name = 'btc_finetune_experiment'
        
        # 模型保存路径
        self.save_path = "/Users/longquan/Documents/git_repository/myself/kronos_zzzz/dataset/outputs/btc_models"
        self.tokenizer_save_folder_name = 'btc_finetune_tokenizer'
        self.predictor_save_folder_name = 'btc_finetune_predictor'
        self.backtest_save_folder_name = 'btc_finetune_backtest'
        
        # 回测结果路径
        self.backtest_result_path = "/Users/longquan/Documents/git_repository/myself/kronos_zzzz/dataset/outputs/btc_backtest_results"
        
        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================
        # 预训练模型路径 (使用HuggingFace Hub)
        self.pretrained_tokenizer_path = "/Users/longquan/Documents/git_repository/myself/kronos_zzzz/NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "/Users/longquan/Documents/git_repository/myself/kronos_zzzz/NeoQuasar/Kronos-base"  # 可以选择small或base
        self.pretrained_tokenizer_path = "/Users/longquan/Documents/git_repository/myself/kronos_zzzz/NeoQuasar/Kronos-Tokenizer-base"
        # 微调后模型路径
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        

        
    def get_data_intervals(self):
        """返回不同阶段的数据时间间隔"""
        return {
            'train': self.train_time_range,
            'val': self.val_time_range, 
            'test': self.test_time_range,
            'backtest': self.backtest_time_range
        }
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("="*60)
        print("🚀 BTC Fine-tuning Configuration")
        print("="*60)
        print(f"📊 Symbol: {self.symbol}")
        print(f"⏰ Data Range: {self.dataset_begin_time} to {self.dataset_end_time}")
        print(f"🎯 Train Range: {self.train_time_range[0]} to {self.train_time_range[1]}")
        print(f"✅ Val Range: {self.val_time_range[0]} to {self.val_time_range[1]}")
        print(f"🧪 Test Range: {self.test_time_range[0]} to {self.test_time_range[1]}")
        print(f"📈 Features: {self.feature_list}")
        print(f"⚙️  Lookback: {self.lookback_window}h, Predict: {self.predict_window}h")
        print(f"🎓 Epochs: {self.epochs}, Batch Size: {self.batch_size}")
        print(f"📚 Pretrained: {self.pretrained_predictor_path}")
        print(f"💾 Save Path: {self.save_path}")
        print("="*60)

if __name__ == '__main__':
    # 测试配置
    config = BTCConfig()
    config.print_config_summary()