# src/training/trainer.py
"""
Trainer v3 — adds three features over v1:

  1. AMP (Automatic Mixed Precision)
     - Uses torch.cuda.amp.autocast + GradScaler when on CUDA.
     - On A100, ~1.7-2x faster training and slightly better generalisation.
     - Falls back to plain FP32 on MPS / CPU.

  2. EMA (Exponential Moving Average) of model weights
     - Maintains a shadow copy of weights with decay 0.9999.
     - Validation and checkpointing use the EMA weights — historically
       gives +0.3-0.8 pp on detection without any extra training cost.

  3. Cleaner inner loop
     - Removed gc.collect() and empty_cache() from inside the train loop
       (they were costing ~5-10% throughput per batch).
     - Done once per epoch instead.

Everything else (mAP eval, save/restore, plotting, label_offset for
RetinaNet) is identical to v1 so the rest of the codebase doesn't change.
"""

import os
import copy
import time
import torch
import numpy as np
from tqdm import tqdm
from src.utils.general import save_checkpoint, ensure_dir


# ─────────────────────────────────────────────────────────────────────
# EMA — keep a smoothed copy of the model parameters
# ─────────────────────────────────────────────────────────────────────
class ModelEMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow model whose params decay-blend toward the live
    model on every optimizer step:
        ema_p = decay * ema_p + (1 - decay) * model_p

    Used for evaluation (smoother weights -> usually higher mAP) and for
    final checkpointing.

    Decay is ramped up from 0 over the first ~2k updates so that the EMA
    isn't biased toward the random initialization at the start.
    """
    def __init__(self, model: torch.nn.Module,
                 decay: float = 0.9999, tau: float = 2000.0):
        # Take a deep copy of the underlying model and put it on the
        # same device. Keep it in eval mode and disable grads — we never
        # backprop through this.
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.updates += 1
        # Ramp-up: prevents EMA from being dominated by random init weights
        d = self.decay * (1 - np.exp(-self.updates / self.tau))

        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


# ─────────────────────────────────────────────────────────────────────
# Visualisation helper (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────
def _save_batch_images(batch_images, batch_targets, save_path,
                      predictions=None, max_images=8):
    """Save a grid of images with bounding boxes drawn (GT or pred)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        n = min(len(batch_images), max_images)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = np.array(axes).reshape(-1) if n > 1 else [axes]

        CLASS_COLORS = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
        ]

        for i in range(len(axes)):
            ax = axes[i]
            ax.axis("off")
            if i >= n:
                continue

            img = batch_images[i]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img  = (img.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(img)

            boxes  = predictions[i]["boxes"].cpu()  if predictions else batch_targets[i]["boxes"].cpu()
            labels = predictions[i]["labels"].cpu() if predictions else batch_targets[i]["labels"].cpu()
            scores = predictions[i].get("scores", torch.ones(len(labels))).cpu() if predictions else None

            for j, (box, lbl) in enumerate(zip(boxes, labels)):
                x1, y1, x2, y2 = box.tolist()
                color = CLASS_COLORS[(lbl.item() - 1) % len(CLASS_COLORS)]
                rect  = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1.5, edgecolor=color, facecolor="none"
                )
                ax.add_patch(rect)
                label_txt = f"{lbl.item()}"
                if scores is not None and j < len(scores):
                    label_txt += f" {scores[j]:.2f}"
                ax.text(x1, y1 - 2, label_txt, color=color,
                        fontsize=6, fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.3, pad=1, linewidth=0))

        plt.tight_layout(pad=0.5)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  [Warning] Could not save batch image: {e}")


# ─────────────────────────────────────────────────────────────────────
# mAP helpers (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────
def _compute_ap(recall, precision):
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        p_at_r = precision[recall >= thr]
        ap += p_at_r.max() if len(p_at_r) else 0.0
    return ap / 101.0


def _ap_per_class(tp, conf, pred_cls, target_cls):
    sort_idx = np.argsort(-conf)
    tp       = tp[:, sort_idx]
    conf     = conf[sort_idx]
    pred_cls = pred_cls[sort_idx]

    unique_cls = np.unique(target_cls)
    ap_list50  = []
    ap_list    = []

    for c in unique_cls:
        mask = pred_cls == c
        n_gt = (target_cls == c).sum()
        if n_gt == 0 or mask.sum() == 0:
            continue

        tp_c   = tp[:, mask]
        n_pred = tp_c.shape[1]
        fpc    = (1 - tp_c).cumsum(1)
        tpc    = tp_c.cumsum(1)

        recall_c    = tpc / (n_gt + 1e-16)
        precision_c = tpc / (tpc + fpc + 1e-16)

        ap_list50.append(_compute_ap(recall_c[0], precision_c[0]))
        ap_list.append(np.mean([
            _compute_ap(recall_c[i], precision_c[i])
            for i in range(tp_c.shape[0])
        ]))

    if not ap_list50:
        return 0.0, 0.0
    return float(np.mean(ap_list50)), float(np.mean(ap_list))


@torch.no_grad()
def evaluate_map(model, dataloader, device, num_classes,
                 iou_thresholds=None, label_offset=0):
    """
    Compute mAP50 / mAP50-95 on `model` (could be the EMA shadow model).

    label_offset same convention as v1:
      SSDLite=0  (labels 1..9, bg at 0)
      RetinaNet=-1 (model uses 0..8 internally, we shift back for compare)
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    n_thr = len(iou_thresholds)
    model.eval()

    all_tp       = []
    all_conf     = []
    all_pred_cls = []
    all_tgt_cls  = []

    from torchvision.ops import box_iou

    for images, targets in dataloader:
        images  = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            pred_boxes  = output["boxes"].cpu()
            pred_scores = output["scores"].cpu()
            pred_labels = (output["labels"].cpu().numpy() - label_offset)

            gt_boxes  = target["boxes"].cpu()
            gt_labels = target["labels"].cpu().numpy()

            all_tgt_cls.extend(gt_labels.tolist())

            if len(pred_boxes) == 0:
                continue

            all_conf.extend(pred_scores.numpy().tolist())
            all_pred_cls.extend(pred_labels.tolist())

            if len(gt_boxes) == 0:
                all_tp.extend([np.zeros(n_thr, dtype=bool)] * len(pred_boxes))
                continue

            iou_matrix = box_iou(pred_boxes, gt_boxes).numpy()
            gt_matched = np.zeros((n_thr, len(gt_boxes)), dtype=bool)

            for pi in range(len(pred_boxes)):
                tp_flags = np.zeros(n_thr, dtype=bool)
                if iou_matrix.shape[1] == 0:
                    all_tp.append(tp_flags)
                    continue

                best_iou_idx = iou_matrix[pi].argmax()
                best_iou     = iou_matrix[pi, best_iou_idx]

                for ti, thr in enumerate(iou_thresholds):
                    if (best_iou >= thr
                            and pred_labels[pi] == gt_labels[best_iou_idx]
                            and not gt_matched[ti, best_iou_idx]):
                        tp_flags[ti]                 = True
                        gt_matched[ti, best_iou_idx] = True

                all_tp.append(tp_flags)

    if not all_conf:
        return 0.0, 0.0

    tp_arr       = np.array(all_tp).T
    conf_arr     = np.array(all_conf)
    pred_cls_arr = np.array(all_pred_cls)
    tgt_cls_arr  = np.array(all_tgt_cls)

    map50, map5095 = _ap_per_class(
        tp_arr, conf_arr, pred_cls_arr, tgt_cls_arr
    )
    return map50, map5095


# ─────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model, optimizer, scheduler,
                 device, config, focal_loss_fn=None):
        self.model         = model
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.device        = device
        self.config        = config
        self.focal_loss_fn = focal_loss_fn

        self.model_name    = config["model_name"]
        self.epochs        = config["epochs"]
        self.patience      = config.get("early_stopping_patience", 10)
        self.warmup_epochs = config.get("warmup_epochs", 3)
        self.base_lr       = config["learning_rate"]
        self.num_classes   = config.get("num_classes", 10)
        self.label_offset  = config.get("label_offset", 0)

        # ── v3 features ────────────────────────────────────────────
        # AMP only on CUDA — MPS doesn't support it cleanly yet
        self.use_amp = (device.type == "cuda")
        self.scaler  = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            print(f"  [Trainer-v3] AMP (mixed precision) ENABLED on {device}")
        else:
            print(f"  [Trainer-v3] AMP disabled (device={device.type})")

        # EMA — enabled by default, can be turned off via config
        self.use_ema = config.get("use_ema", True)
        if self.use_ema:
            self.ema = ModelEMA(model, decay=0.9999, tau=2000.0)
            print(f"  [Trainer-v3] EMA enabled (decay=0.9999, ramp tau=2000)")
        else:
            self.ema = None

        self.checkpoints_dir = config["output"]["checkpoints_dir"]
        self.logs_dir        = config["output"]["logs_dir"]
        ensure_dir(self.checkpoints_dir)
        ensure_dir(self.logs_dir)

        self.best_map50        = 0.0
        self.epochs_no_improve = 0
        self.history = {
            "train_loss": [],
            "val_loss":   [],
            "map50":      [],
            "map50_95":   [],
            "epoch_time": [],
            "lr":         [],
        }

    # ── helpers ────────────────────────────────────────────────────
    def _warmup_lr(self, epoch):
        if epoch <= self.warmup_epochs:
            warmup_factor = 0.1 + 0.9 * (epoch / self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.base_lr * warmup_factor

    def _get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def _prep_targets(self, targets):
        prepped = []
        for t in targets:
            t2 = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            if self.label_offset != 0 and "labels" in t2 and isinstance(t2["labels"], torch.Tensor):
                t2["labels"] = t2["labels"] + self.label_offset
            prepped.append(t2)
        return prepped

    def _eval_model(self):
        """Return the model to use for evaluation — EMA shadow if enabled."""
        return self.ema.module if self.ema is not None else self.model

    # ── train one epoch ────────────────────────────────────────────
    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss  = 0.0
        num_batches = 0

        loop = tqdm(dataloader, desc="  Training", leave=False)

        for images, targets in loop:
            images  = [img.to(self.device, non_blocking=True) for img in images]
            targets = self._prep_targets(targets)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    loss_dict = self.model(images, targets)
                    losses    = sum(loss for loss in loss_dict.values())
                self.scaler.scale(losses).backward()
                # Unscale BEFORE clipping so the threshold is in real-loss units
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict = self.model(images, targets)
                losses    = sum(loss for loss in loss_dict.values())
                losses.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
                )
                self.optimizer.step()

            # Update EMA after the optimiser step
            if self.ema is not None:
                self.ema.update(self.model)

            total_loss  += losses.item()
            num_batches += 1
            loop.set_postfix(loss=f"{losses.item():.4f}")

        # Periodic cleanup — moved out of the inner loop
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

        return total_loss / max(num_batches, 1)

    # ── validate one epoch ─────────────────────────────────────────
    @torch.no_grad()
    def validate_one_epoch(self, dataloader):
        # Important: torchvision detection models only return loss in
        # train() mode. EMA shadow stays in eval mode for predictions,
        # but for the loss curve we keep using the live model.
        self.model.train()
        total_loss  = 0.0
        num_batches = 0

        loop = tqdm(dataloader, desc="  Validating", leave=False)

        for images, targets in loop:
            images  = [img.to(self.device, non_blocking=True) for img in images]
            targets = self._prep_targets(targets)

            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    loss_dict = self.model(images, targets)
                    losses    = sum(loss for loss in loss_dict.values())
            else:
                loss_dict = self.model(images, targets)
                losses    = sum(loss for loss in loss_dict.values())

            total_loss  += losses.item()
            num_batches += 1
            loop.set_postfix(loss=f"{losses.item():.4f}")

        return total_loss / max(num_batches, 1)

    # ── main fit ───────────────────────────────────────────────────
    def fit(self, train_loader, val_loader):
        print(f"\n{'='*55}")
        print(f"  Training   : {self.model_name}")
        print(f"  Epochs     : {self.epochs}")
        print(f"  Device     : {self.device}")
        print(f"  Base LR    : {self.base_lr}")
        print(f"  Warmup     : {self.warmup_epochs} epochs")
        print(f"  Patience   : {self.patience}")
        print(f"  AMP        : {self.use_amp}")
        print(f"  EMA        : {self.use_ema}")
        print(f"{'='*55}")

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            self._warmup_lr(epoch)
            current_lr = self._get_current_lr()

            print(f"\nEpoch [{epoch}/{self.epochs}] | LR: {current_lr:.6f}")

            train_loss = self.train_one_epoch(train_loader)

            if epoch == 1:
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (imgs, tgts) in enumerate(train_loader):
                        if batch_idx >= 3:
                            break
                        save_path = os.path.join(
                            self.logs_dir, f"train_batch{batch_idx}.jpg"
                        )
                        _save_batch_images(imgs, tgts, save_path)
                self.model.train()

            val_loss = self.validate_one_epoch(val_loader)

            # mAP eval — uses EMA model if enabled
            map50, map5095 = evaluate_map(
                self._eval_model(),
                val_loader, self.device, self.num_classes,
                label_offset=self.label_offset,
            )

            if epoch == 1:
                eval_model = self._eval_model()
                eval_model.eval()
                with torch.no_grad():
                    for batch_idx, (imgs, tgts) in enumerate(val_loader):
                        if batch_idx >= 2:
                            break
                        _save_batch_images(
                            imgs, tgts,
                            os.path.join(self.logs_dir, f"val_batch{batch_idx}_labels.jpg")
                        )
                        preds = eval_model([img.to(self.device) for img in imgs])
                        _save_batch_images(
                            imgs, tgts,
                            os.path.join(self.logs_dir, f"val_batch{batch_idx}_pred.jpg"),
                            predictions=preds
                        )
                self.model.train()

            if epoch > self.warmup_epochs:
                self.scheduler.step()

            epoch_time = time.time() - start_time
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["map50"].append(map50)
            self.history["map50_95"].append(map5095)
            self.history["epoch_time"].append(epoch_time)
            self.history["lr"].append(current_lr)

            print(
                f"  Train Loss : {train_loss:.4f} | "
                f"Val Loss : {val_loss:.4f} | "
                f"mAP50 : {map50:.4f} | "
                f"mAP50-95 : {map5095:.4f} | "
                f"Time : {epoch_time:.1f}s"
            )

            # Checkpoint best mAP50 — save the EMA weights when EMA is on
            if map50 > self.best_map50:
                self.best_map50        = map50
                self.epochs_no_improve = 0
                ckpt_model = self._eval_model()
                ckpt_path = os.path.join(
                    self.checkpoints_dir,
                    f"{self.model_name}_best.pth"
                )
                save_checkpoint(
                    ckpt_model, self.optimizer, epoch, ckpt_path,
                    extra={
                        "map50":      map50,
                        "map50_95":   map5095,
                        "val_loss":   val_loss,
                        "model_name": self.model_name,
                        "is_ema":     self.use_ema,
                    }
                )
            else:
                self.epochs_no_improve += 1
                print(f"  No improvement for "
                      f"{self.epochs_no_improve} epoch(s).")

            if self.epochs_no_improve >= self.patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

        print(f"\n  Training complete.")
        print(f"  Best mAP50 : {self.best_map50:.4f}")
        self._save_history()
        self._save_plots()
        return self.history

    # ── persistence helpers ────────────────────────────────────────
    def _save_history(self):
        import pandas as pd
        df   = pd.DataFrame(self.history)
        path = os.path.join(
            self.logs_dir, f"{self.model_name}_history.csv"
        )
        df.to_csv(path, index=False)
        print(f"  History saved to {path}")

    def _save_plots(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = range(1, len(self.history["train_loss"]) + 1)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"{self.model_name} — Training Curves", fontsize=14)

            axes[0, 0].plot(epochs, self.history["train_loss"], label="Train Loss")
            axes[0, 0].plot(epochs, self.history["val_loss"],   label="Val Loss")
            axes[0, 0].set_title("Loss");      axes[0, 0].legend(); axes[0, 0].grid(True)

            axes[0, 1].plot(epochs, self.history["map50"], color="green")
            axes[0, 1].set_title("mAP50");     axes[0, 1].set_ylim(0, 1); axes[0, 1].grid(True)

            axes[1, 0].plot(epochs, self.history["map50_95"], color="orange")
            axes[1, 0].set_title("mAP50-95");  axes[1, 0].set_ylim(0, 1); axes[1, 0].grid(True)

            axes[1, 1].plot(epochs, self.history["lr"], color="red")
            axes[1, 1].set_title("Learning Rate"); axes[1, 1].grid(True)

            plt.tight_layout()
            plot_path = os.path.join(
                self.logs_dir, f"{self.model_name}_curves.png"
            )
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  Plots saved to {plot_path}")
        except Exception as e:
            print(f"  [Warning] Could not save plots: {e}")
