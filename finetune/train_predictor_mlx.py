# train_predictor_mlx.py
import os
import sys
import json
import time
from time import gmtime, strftime
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

# Ensure project root is in path
sys.path.append('../')
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer, Kronos
# Import shared utilities
from utils.training_utils import set_seed, get_model_size, format_time

def create_dataloaders(config: dict):
    """
    Creates and returns dataloaders for training and validation.

    Args:
        config (dict): A dictionary of configuration parameters.

    Returns:
        tuple: (train_loader, val_loader, train_dataset, valid_dataset).
    """
    print("Creating dataloaders...")
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    # For MLX, we'll use simple data iteration instead of DataLoader
    return train_dataset, valid_dataset, train_dataset, valid_dataset

def move_to_device(batch_x, batch_x_stamp, device=None):
    """
    Move data to device (MLX uses unified memory, so this is more about ensuring correct format)
    """
    # In MLX, data is typically kept in unified memory, but we ensure correct format
    batch_x = mx.array(batch_x.numpy() if hasattr(batch_x, 'numpy') else batch_x)
    batch_x_stamp = mx.array(batch_x_stamp.numpy() if hasattr(batch_x_stamp, 'numpy') else batch_x_stamp)
    return batch_x, batch_x_stamp

def compute_loss_fn(model, token_in_0, token_in_1, token_out_0, token_out_1, stamp_data):
    """
    Compute loss for the model
    """
    logits = model(token_in_0, token_in_1, stamp_data)
    # 根据PyTorch版本，损失计算在model.head中
    loss, s1_loss, s2_loss = model.head.compute_loss(logits[0], logits[1], token_out_0, token_out_1)
    return loss, s1_loss, s2_loss

def train_model(model, tokenizer, config, save_dir, logger):
    """
    The main training and validation loop for the predictor using MLX.
    """
    start_time = time.time()
    effective_bs = config['batch_size']
    print(f"BATCHSIZE: {config['batch_size']}, Total: {effective_bs}")

    train_dataset, val_dataset, _, _ = create_dataloaders(config)

    # MLX optimizer
    optimizer = optim.AdamW(
        learning_rate=config['predictor_learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay']
    )

    # MLX loss and state - 使用自定义损失函数
    loss_and_grad_fn = nn.value_and_grad(model, compute_loss_fn)

    best_val_loss = float('inf')
    dt_result = {}
    batch_idx_global = 0

    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()

        # Reset dataset for new epoch
        train_dataset.set_epoch_seed(epoch_idx * 10000)

        total_loss = 0.0
        num_batches = 0

        for i, (batch_x, batch_x_stamp) in enumerate(train_dataset):
            # Prepare batch data
            batch_x, batch_x_stamp = move_to_device(batch_x, batch_x_stamp)

            # Tokenize input data on-the-fly
            token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            # Prepare inputs and targets for the language model
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward pass and loss calculation with gradient
            (loss, s1_loss, s2_loss), grads = loss_and_grad_fn(
                model, token_in[0], token_in[1], token_out[0], token_out[1], batch_x_stamp[:, :-1, :]
            )

            # Update model parameters
            optimizer.update(model, grads)

            # Ensure parameter updates are applied
            mx.eval(model.parameters())
            mx.eval(optimizer.state)

            total_loss += loss.item()
            num_batches += 1

            # Logging
            if (batch_idx_global + 1) % config['log_interval'] == 0:
                print(
                    f"[Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}] "
                    f"Loss: {loss.item():.4f}"
                )
                if logger:
                    logger.log_metric('train_predictor_loss_batch', loss.item(), step=batch_idx_global)
                    logger.log_metric('train_S1_loss_each_batch', s1_loss.item(), step=batch_idx_global)
                    logger.log_metric('train_S2_loss_each_batch', s2_loss.item(), step=batch_idx_global)

            batch_idx_global += 1

        # --- Validation Loop ---
        model.eval()
        tot_val_loss_sum = 0.0
        val_batches_processed = 0

        val_dataset.set_epoch_seed(0)

        for batch_x, batch_x_stamp in val_dataset:
            batch_x, batch_x_stamp = move_to_device(batch_x, batch_x_stamp)

            token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Validation 不需要梯度计算
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            val_loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            tot_val_loss_sum += val_loss.item()
            val_batches_processed += 1

        avg_val_loss = tot_val_loss_sum / val_batches_processed if val_batches_processed > 0 else 0

        # --- End of Epoch Summary & Checkpointing ---
        print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
        print(f"Average Training Loss: {total_loss / num_batches:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
        print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")

        if logger:
            logger.log_metric('val_predictor_loss_epoch', avg_val_loss, epoch=epoch_idx)
            logger.log_metric('train_predictor_loss_epoch', total_loss / num_batches, epoch=epoch_idx)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"{save_dir}/checkpoints/best_model"
            # Save model in MLX format
            model.save_weights(save_path)
            print(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")

    dt_result['best_val_loss'] = best_val_loss
    return dt_result

def main(config: dict):
    """Main function to orchestrate the MLX training process."""
    # For MLX, we don't need DDP setup
    rank = 0  # Single device training

    # Device selection (MLX automatically uses Metal on macOS)
    device = mx.default_device()
    print(f"Using device: {device}")

    set_seed(config['seed'], rank)

    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])

    # Logger and summary setup
    logger, master_summary = None, {}
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    master_summary = {
        'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        'save_directory': save_dir,
    }

    # Model Initialization
    tokenizer = KronosTokenizer.from_pretrained(config['finetuned_tokenizer_path'])
    tokenizer.eval()

    model = Kronos.from_pretrained(config['pretrained_predictor_path'])
    model.train()  # Set to training mode

    print(f"Predictor Model Size: {get_model_size(model)}")

    # Start Training
    dt_result = train_model(
        model, tokenizer, config, save_dir, logger
    )

    master_summary['final_result'] = dt_result
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(master_summary, f, indent=4)
    print('Training finished. Summary file saved.')
    if logger:
        logger.end()

if __name__ == '__main__':
    config_instance = Config()
    main(config_instance.__dict__)
