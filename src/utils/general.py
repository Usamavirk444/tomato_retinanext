# src/utils/general.py

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Seed every RNG we touch so training runs are reproducible across
    the 3-seed protocol (42, 123, 456). Notes for the paper:
      * PYTHONHASHSEED is set *before* torch is imported in train_all.py
        so hash-based ops (e.g. set iteration) match across runs.
      * MPS determinism is only partial — we seed it where we can, but
        Apple's backend still has some non-determinism in reductions.
        Document this in the methods section.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def collate_fn(batch):
    return tuple(zip(*batch))


def load_model_for_eval(model, checkpoint_path: str, device):
    """
    Load checkpoint into model for evaluation.

    Args:
        model:           PyTorch model (already built with correct architecture)
        checkpoint_path: Path to .pth checkpoint file
        device:          torch.device

    Returns:
        model, epoch, val_loss
    """
    print(f"\nCheckpoint loaded")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])

    epoch    = ckpt.get("epoch", -1)
    val_loss = ckpt.get("val_loss", -1)

    print(f"Epoch      : {epoch}")
    print(f"Val Loss   : {val_loss}")
    print(f"Model ready for evaluation\n")

    return model, epoch, val_loss


def save_checkpoint(model, optimizer, epoch: int, path: str, extra: dict = None):
    """Save model + optimizer checkpoint."""
    ensure_dir(os.path.dirname(path))
    state = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        state.update(extra)
    torch.save(state, path)
    print(f"[Checkpoint] Saved to {path} (epoch {epoch})")
