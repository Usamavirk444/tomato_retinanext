"""
Inference-speed benchmark for all four models.

Reports: mean ± std inference time (ms), throughput (FPS), and parameter count.
All numbers measured at batch_size=1 with 640x640 input on whatever device
is available (CUDA preferred). Uses 10 warmup iterations and 100 timed
iterations. The 100 iterations is enough for sub-millisecond stability on
modern GPUs.

Usage:
    python benchmark_fps.py
    python benchmark_fps.py --batch-size 1 --warmup 10 --iters 100
    python benchmark_fps.py --seed 42         # specific seed (default uses seed 42)
    python benchmark_fps.py --device cuda     # force device

Output:
    Prints a comparison table to stdout
    Saves CSV at outputs/results/fps_benchmark.csv
"""

import os
import sys
import time
import argparse
import csv

import torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# moved from paper_files/ into scripts/eval/ - walk up to the repo root
PROJECT_ROOT = HERE
while not os.path.exists(os.path.join(PROJECT_ROOT, "train_all.py")) \
        and os.path.dirname(PROJECT_ROOT) != PROJECT_ROOT:
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def count_params_M(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_torchvision_model(model, device, batch_size, warmup, iters,
                              img_size=640):
    """Measure inference time for a torchvision-style detection model.

    Note: torchvision detection models accept a list of tensors in eval mode,
    not a stacked batch tensor. We feed `batch_size` identical tensors per call.
    """
    model.eval()
    dummy = torch.randn(3, img_size, img_size, device=device)
    inputs = [dummy.clone() for _ in range(batch_size)]

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    with torch.no_grad():
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times_ms)
    return times


def measure_yolov8(weights_path, device, batch_size, warmup, iters,
                   img_size=640):
    """Measure inference time for a YOLOv8 model via Ultralytics."""
    from ultralytics import YOLO
    model = YOLO(weights_path)

    # Build a dummy numpy image (HWC, uint8) since Ultralytics accepts this
    dummy_np = (np.random.rand(img_size, img_size, 3) * 255).astype(np.uint8)

    # Warmup
    for _ in range(warmup):
        _ = model.predict(dummy_np, device=device.index if device.type == "cuda" else "cpu",
                          verbose=False, imgsz=img_size)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.predict(dummy_np, device=device.index if device.type == "cuda" else "cpu",
                          verbose=False, imgsz=img_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times_ms)
    params_M = sum(p.numel() for p in model.model.parameters()) / 1e6
    return times, params_M


def load_torchvision_model(name, seed, device):
    """Load a torchvision-style model from its 'best' checkpoint."""
    if name == "ssdlite":
        from src.models.ssdlite import build_ssdlite_vanilla
        ckpt = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "ssdlite",
                            f"seed_{seed}",
                            "ssdlite_mobilenet_v3_large_best.pth")
        model = build_ssdlite_vanilla(num_classes=10, pretrained=False)
    elif name == "retinanet":
        from src.models.retinanet import build_retinanet
        ckpt = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "retinanet",
                            f"seed_{seed}",
                            "retinanet_resnet50_fpn_v2_best.pth")
        model = build_retinanet(num_classes=10, pretrained=False)
    elif name == "retinanet_hybrid":
        from src.models.hybrid import build_retinanet_hybrid
        ckpt = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "retinanet_hybrid",
                            f"seed_{seed}",
                            "retinanet_hybrid_resnet50_bifpn_cbam_best.pth")
        model = build_retinanet_hybrid(num_classes=10, pretrained=False)
    else:
        raise ValueError(name)

    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd.get("model", sd))
        model.load_state_dict(sd)
        print(f"  Loaded checkpoint: {ckpt}")
    else:
        print(f"  [WARN] No checkpoint at {ckpt} — using random init weights "
              f"(speed numbers are still valid)")
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--img-size", type=int, default=640)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Warmup iters : {args.warmup}")
    print(f"Timed iters  : {args.iters}")
    print(f"Image size   : {args.img_size}x{args.img_size}")
    print()

    rows = []
    models_to_test = [
        ("ssdlite",          "SSDLite-MobileNetV3-Large"),
        ("retinanet",        "RetinaNet-ResNet50-FPN-v2"),
        ("retinanet_hybrid", "RetinaNet-Hybrid v3"),
    ]

    # ── torchvision models ──
    for key, label in models_to_test:
        print(f"Benchmarking {label} ...")
        try:
            model = load_torchvision_model(key, args.seed, device)
            params_M = count_params_M(model)
            times = measure_torchvision_model(
                model, device,
                batch_size=args.batch_size,
                warmup=args.warmup,
                iters=args.iters,
                img_size=args.img_size,
            )
            mean_ms = times.mean()
            std_ms = times.std()
            fps = 1000.0 / mean_ms * args.batch_size
            rows.append((label, params_M, mean_ms, std_ms, fps))
            print(f"  Params: {params_M:.2f} M  |  "
                  f"Time/iter: {mean_ms:.2f} ± {std_ms:.2f} ms  |  "
                  f"FPS: {fps:.1f}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [FAIL] {e}")
            rows.append((label, 0.0, 0.0, 0.0, 0.0))

    # ── YOLOv8n ──
    print("Benchmarking YOLOv8n ...")
    try:
        weights = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "yolov8n",
                               f"seed_{args.seed}", "weights", "best.pt")
        if not os.path.exists(weights):
            # Fallback to project-root weights file
            weights = os.path.join(PROJECT_ROOT, "yolov8n.pt")
            print(f"  No trained best.pt; using {weights}")
        times, params_M = measure_yolov8(
            weights, device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iters=args.iters,
            img_size=args.img_size,
        )
        mean_ms = times.mean()
        std_ms = times.std()
        fps = 1000.0 / mean_ms * args.batch_size
        rows.append(("YOLOv8n", params_M, mean_ms, std_ms, fps))
        print(f"  Params: {params_M:.2f} M  |  "
              f"Time/iter: {mean_ms:.2f} ± {std_ms:.2f} ms  |  "
              f"FPS: {fps:.1f}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        rows.append(("YOLOv8n", 0.0, 0.0, 0.0, 0.0))

    # ── Print summary table ──
    print()
    print("="*80)
    print(f"  {'Model':<32} {'Params (M)':>11} {'Time (ms)':>14} {'FPS':>10}")
    print("="*80)
    for label, params_M, mean_ms, std_ms, fps in rows:
        time_str = f"{mean_ms:.2f} ± {std_ms:.2f}" if mean_ms > 0 else "—"
        print(f"  {label:<32} {params_M:>11.2f} {time_str:>14} {fps:>10.1f}")
    print("="*80)

    # ── Save CSV ──
    out_path = os.path.join(PROJECT_ROOT, "outputs", "results",
                            "fps_benchmark.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "params_M", "time_ms_mean", "time_ms_std", "fps",
                    "device", "batch_size", "img_size"])
        for label, params_M, mean_ms, std_ms, fps in rows:
            w.writerow([label, params_M, mean_ms, std_ms, fps,
                        str(device), args.batch_size, args.img_size])
    print(f"\nSaved CSV: {out_path}")


if __name__ == "__main__":
    main()
