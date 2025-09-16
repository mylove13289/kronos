import os
import random
import datetime
import numpy as np
import torch
import torch.distributed as dist


# 在 utils.training_utils.py 中修改 setup_ddp 函数
def setup_ddp():
    """
    Initialize the distributed training environment.
    """
    # 获取本地rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 明确设置设备ID
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}")  # 明确指定device_id
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 设置当前设备
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_ddp():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0):
    """
    Sets the random seed for reproducibility across all relevant libraries.

    Args:
        seed (int): The base seed value.
        rank (int): The process rank, used to ensure different processes have
                    different seeds, which can be important for data loading.
    """
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
        # The two lines below can impact performance, so they are often
        # reserved for final experiments where reproducibility is critical.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_size(model: torch.nn.Module) -> str:
    """
    Calculates the number of trainable parameters in a PyTorch model and returns
    it as a human-readable string.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        str: A string representing the model size (e.g., "175.0B", "7.1M", "50.5K").
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"  # Billions
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"  # Millions
    else:
        return f"{total_params / 1e3:.1f}K"  # Thousands


def reduce_tensor(tensor: torch.Tensor, world_size: int, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    Reduces a tensor's value across all processes in a distributed setup.

    Args:
        tensor (torch.Tensor): The tensor to be reduced.
        world_size (int): The total number of processes.
        op (dist.ReduceOp, optional): The reduction operation (SUM, AVG, etc.).
                                      Defaults to dist.ReduceOp.SUM.

    Returns:
        torch.Tensor: The reduced tensor, which will be identical on all processes.
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    # Note: `dist.ReduceOp.AVG` is available in newer torch versions.
    # For compatibility, manual division is sometimes used after a SUM.
    if op == dist.ReduceOp.AVG:
        rt /= world_size
    return rt


def format_time(seconds: float) -> str:
    """
    Formats a duration in seconds into a human-readable H:M:S string.

    Args:
        seconds (float): The total seconds.

    Returns:
        str: The formatted time string (e.g., "0:15:32").
    """
    return str(datetime.timedelta(seconds=int(seconds)))



