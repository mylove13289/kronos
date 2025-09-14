import os

class Config:
    """
    Configuration class for the entire project.
    """

    def __init__(self):
        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        # TODO: Update this path to your Qlib data directory.
        self.qlib_data_path = "~/.qlib/qlib_data/cn_data"
        self.instrument = 'csi300'

        # BTC数据参数
        self.symbol = 'BTCUSDT'
        self.data_source = 'binance'  # 使用Binance数据源

        # Overall time range for data loading from Qlib.
        self.dataset_begin_time = "2109-01-01"
        self.dataset_end_time = '2025-09-01'

        # Sliding window parameters for creating samples.
        self.lookback_window = 90  # Number of past time steps for input.
        self.predict_window = 10  # Number of future time steps for prediction.
        self.max_context = 512  # Maximum context length for the model.

        # Features to be used from the raw data.
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        # Time-based features to be generated.
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # =================================================================
        # Dataset Splitting & Paths
        # =================================================================
        # Note: The validation/test set starts earlier than the training/validation set ends
        # to account for the `lookback_window`.
        self.train_time_range = [self.dataset_begin_time, self.dataset_end_time]
        self.val_time_range = [self.dataset_begin_time, self.dataset_end_time]
        self.test_time_range = [self.dataset_begin_time, self.dataset_end_time]
        self.backtest_time_range = [self.dataset_begin_time, self.dataset_end_time]
        #self.use_comet = False

        # TODO: Directory to save the processed, pickled datasets.
        self.dataset_path = "/Users/longquan/Documents/git_repository/myself/kronos/data/processed_datasets"

        # =================================================================
        # Training Hyperparameters
        # =================================================================
        self.clip = 5.0  # Clipping value for normalized data to prevent outliers.
        """
        定义训练轮数：
        epochs
        指定了完整遍历训练数据集的次数
        在你的配置中设置为
        30，意味着模型将对整个训练数据集进行30次完整的训练
        控制训练时长：
        每个
        epoch
        包含对所有训练样本的一次完整处理
        更多的
        epochs
        通常意味着更长的训练时间和可能更好的模型性能（但也可能导致过拟合）
        在训练循环中的使用："""
        self.epochs = 15
        self.log_interval = 100  # Log training status every N batches.
        self.batch_size = 50  # Batch size per GPU.

        # Number of samples to draw for one "epoch" of training/validation.
        # This is useful for large datasets where a true epoch is too long.
        #self.n_train_iter = 2000 * self.batch_size
        self.n_train_iter = 800000
        self.n_val_iter = 400 * self.batch_size

        # Learning rates for different model components.
        # 学习率 (BTC数据需要更小的学习率)
        self.tokenizer_learning_rate = 4e-4
        # self.predictor_learning_rate = 2e-5
        self.predictor_learning_rate = 8e-4

        # Gradient accumulation to simulate a larger batch size.
        self.accumulation_steps = 1

        # AdamW optimizer parameters.
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1

        # Miscellaneous
        self.seed = 100  # Global random seed for reproducibility.

        # =================================================================
        # Experiment Logging & Saving
        # =================================================================
        self.use_comet = False # Set to False if you don't want to use Comet ML
        self.comet_config = {
            # It is highly recommended to load secrets from environment variables
            # for security purposes. Example: os.getenv("COMET_API_KEY")
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-Finetune-Demo",
            "workspace": "your_comet_workspace" # TODO: Change to your Comet ML workspace name
        }
        self.comet_tag = 'finetune_demo'
        self.comet_name = 'finetune_demo'

        # Base directory for saving model checkpoints and results.
        # Using a general 'outputs' directory is a common practice.
        self.save_path = "/Users/longquan/Documents/git_repository/myself/kronos/data/outputs/models"
        self.tokenizer_save_folder_name = 'finetune_tokenizer_demo'
        self.predictor_save_folder_name = 'finetune_predictor_demo'
        self.backtest_save_folder_name = 'finetune_backtest_demo'

        # Path for backtesting results.
        self.backtest_result_path = "/Users/longquan/Documents/git_repository/myself/kronos/data/outputs/backtest_results"

        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================
        # TODO: Update these paths to your pretrained model locations.
        # These can be local paths or Hugging Face Hub model identifiers.
        #self.pretrained_tokenizer_path = "/home/admin/software/NeoQuasar/Kronos-Tokenizer-base"
        #self.pretrained_predictor_path = "/home/admin/software/NeoQuasar/Kronos-small"

        self.pretrained_tokenizer_path = "/Users/longquan/Documents/MYSELF/models/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "/Users/longquan/Documents/MYSELF/models/Kronos-base"

        # Paths to the fine-tuned models, derived from the save_path.
        # These will be generated automatically during training.
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        # =================================================================
        # Backtesting Parameters
        # =================================================================
        self.backtest_n_symbol_hold = 50  # Number of symbols to hold in the portfolio.
        self.backtest_n_symbol_drop = 5  # Number of symbols to drop from the pool.
        self.backtest_hold_thresh = 5  # Minimum holding period for a stock.
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = self._set_benchmark(self.instrument)

    def _set_benchmark(self, instrument):
        dt_benchmark = {
            'csi800': "SH000906",
            'csi1000': "SH000852",
            'csi300': "SH000300",
        }
        if instrument in dt_benchmark:
            return dt_benchmark[instrument]
        else:
            raise ValueError(f"Benchmark not defined for instrument: {instrument}")
