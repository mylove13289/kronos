import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from btc_config import BTCConfig


class BTCDataset(Dataset):
    """
    A PyTorch Dataset for handling BTC financial time series data.

    This dataset pre-computes all possible start indices for sliding windows
    and then randomly samples from them during training/validation.

    Args:
        data_type (str): The type of dataset to load, either 'train' or 'val'.

    Raises:
        ValueError: If `data_type` is not 'train' or 'val'.
    """

    def __init__(self, data_type: str = 'train'):
        self.config = BTCConfig()
        if data_type not in ['train', 'val', 'test', 'backtest']:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type

        # Use a dedicated random number generator for sampling to avoid
        # interfering with other random processes (e.g., in model initialization).
        self.py_rng = random.Random(self.config.seed)
        
        # 添加过拟合实验支持
        self.overfit_mode = False
        self.overfit_indices = None
        self.fixed_samples = None

        # Set paths and number of samples based on the data type.
        if data_type == 'train':
            self.data_path = f"{self.config.dataset_path}/train_data.pkl"
            self.n_samples = self.config.n_train_iter
        elif data_type == 'val':
            self.data_path = f"{self.config.dataset_path}/val_data.pkl"
            self.n_samples = self.config.n_val_iter
        elif data_type == 'test':
            self.data_path = f"{self.config.dataset_path}/test_data.pkl"
            self.n_samples = 1000
        elif data_type == 'backtest':
            self.data_path = f"{self.config.dataset_path}/backtest_data.pkl"
            self.n_samples = 1000

        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.window = self.config.lookback_window + self.config.predict_window + 1

        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list

        # Pre-compute all possible (symbol, start_index) pairs.
        self.indices = []
        print(f"[{data_type.upper()}] Pre-computing sample indices...")
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            series_len = len(df)
            num_samples = series_len - self.window + 1

            if num_samples > 0:
                # Generate time features and store them directly in the dataframe.
                # 重命名datetime列（从reset_index得来的）
                if 'timestamp' in df.columns:
                    df['datetime'] = df['timestamp']
                elif df.index.name == 'timestamp' or 'datetime' not in df.columns:
                    # 如果没有datetime列，使用索引
                    df['datetime'] = df.index
                
                # 确保datetime是pandas datetime类型
                df['datetime'] = pd.to_datetime(df['datetime'])
                
                # 生成时间特征
                df['minute'] = df['datetime'].dt.minute
                df['hour'] = df['datetime'].dt.hour
                df['weekday'] = df['datetime'].dt.weekday
                df['day'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                
                # Keep only necessary columns to save memory.
                available_features = [col for col in self.feature_list if col in df.columns]
                if len(available_features) != len(self.feature_list):
                    missing = set(self.feature_list) - set(available_features)
                    print(f"Warning: Missing features {missing} for symbol {symbol}")
                    # 继续使用可用的特征
                    self.feature_list = available_features
                
                # 确保使用所有可用的特征，包括amount
                final_features = [col for col in df.columns if col in self.feature_list or col == 'amount']
                self.feature_list = final_features
                
                self.data[symbol] = df[self.feature_list + self.time_feature_list]

                # Add all valid starting indices for this symbol to the global list.
                for i in range(num_samples):
                    self.indices.append((symbol, i))

        # The effective dataset size is the minimum of the configured iterations
        # and the total number of available samples.
        self.n_samples = min(self.n_samples, len(self.indices))
        print(f"[{data_type.upper()}] Found {len(self.indices)} possible samples. Using {self.n_samples} per epoch.")

    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch. This is crucial
        for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def enable_overfit_mode(self, num_samples: int = 32):
        """
        启用过拟合模式，固定使用指定数量的样本
        
        Args:
            num_samples (int): 要使用的固定样本数量
        """
        self.overfit_mode = True
        # 选择前 num_samples 个索引作为固定样本
        self.overfit_indices = self.indices[:min(num_samples, len(self.indices))]
        
        # 预先计算这些固定样本，避免每次重新计算
        self.fixed_samples = []
        for i, (symbol, start_idx) in enumerate(self.overfit_indices):
            df = self.data[symbol]
            end_idx = start_idx + self.window
            win_df = df.iloc[start_idx:end_idx]
            
            x = win_df[self.feature_list].values.astype(np.float32)
            x_stamp = win_df[self.time_feature_list].values.astype(np.float32)
            
            # 实例级归一化
            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_normalized = (x - x_mean) / (x_std + 1e-5)
            x_normalized = np.clip(x_normalized, -self.config.clip, self.config.clip)
            
            x_tensor = torch.from_numpy(x_normalized)
            x_stamp_tensor = torch.from_numpy(x_stamp)
            x_mean_tensor = torch.from_numpy(x_mean.astype(np.float32))
            x_std_tensor = torch.from_numpy(x_std.astype(np.float32))
            
            # 过拟合模式也保存反标准化参数，以便验证时使用
            self.fixed_samples.append({
                'x': x_tensor,
                'x_stamp': x_stamp_tensor,
                'x_mean': x_mean_tensor,
                'x_std': x_std_tensor
            })
        
        print(f"Overfit mode enabled with {len(self.fixed_samples)} fixed samples")

    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        if self.overfit_mode and self.fixed_samples:
            return len(self.fixed_samples)
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retrieves a sample from the dataset.
        
        Sampling Strategy:
        - Training set ('train'): Random sampling for data diversity
        - Validation set ('val'): Deterministic sampling for reproducibility
        - Overfit mode: Fixed pre-computed samples
        
        Args:
            idx (int): Sample index (ignored for train set random sampling)
            
        Returns:
            A dictionary containing tensors for the sample.
            For training: {'x': tensor, 'x_stamp': tensor}
            For val/test/backtest: {'x': ..., 'x_stamp': ..., 'x_mean': ..., 'x_std': ...}
        """
        if self.overfit_mode and self.fixed_samples:
            # 过拟合模式：使用固定样本，允许重复
            actual_idx = idx % len(self.fixed_samples)
            # 过拟合模式现在也返回反标准化参数（与验证模式一致）
            return self.fixed_samples[actual_idx]
        
        # 根据数据集类型选择采样策略
        if self.data_type in ['val', 'test', 'backtest']:
            # 验证集：确定性采样，保证可复现性
            deterministic_idx = idx % len(self.indices)
            symbol, start_idx = self.indices[deterministic_idx]
        else:
            # 训练集：随机采样，增加数据多样性
            random_idx = self.py_rng.randint(0, len(self.indices) - 1)
            symbol, start_idx = self.indices[random_idx]

        # Extract the sliding window from the dataframe.
        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        # Separate main features and time features.
        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # Perform instance-level normalization.
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_normalized = (x - x_mean) / (x_std + 1e-5)
        x_normalized = np.clip(x_normalized, -self.config.clip, self.config.clip)

        # Convert to PyTorch tensors and build the batch dictionary.
        batch = {
            'x': torch.from_numpy(x_normalized),
            'x_stamp': torch.from_numpy(x_stamp),
        }
        
        # For non-training modes, also return normalization parameters for precise denormalization
        if self.data_type in ['val', 'test', 'backtest']:
            batch['x_mean'] = torch.from_numpy(x_mean.astype(np.float32))
            batch['x_std'] = torch.from_numpy(x_std.astype(np.float32))
        
        return batch


if __name__ == '__main__':
    # Example usage and verification.
    import pandas as pd
    
    print("Creating BTC training dataset instance...")
    train_dataset = BTCDataset(data_type='train')

    print(f"Dataset length: {len(train_dataset)}")

    if len(train_dataset) > 0:
        sample_dict = train_dataset[100]  # Index 100 is ignored.
        try_x = sample_dict['x']
        try_x_stamp = sample_dict['x_stamp']
        print(f"Sample feature shape: {try_x.shape}")
        print(f"Sample time feature shape: {try_x_stamp.shape}")
        print(f"Sample feature tensor:")
        print(try_x[:5])  # 显示前5个时间步
        print(f"Sample time feature tensor:")
        print(try_x_stamp[:5])  # 显示前5个时间步
    else:
        print("Dataset is empty.")