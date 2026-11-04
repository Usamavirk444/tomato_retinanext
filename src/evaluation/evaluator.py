# src/evaluation/evaluator.py

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.utils.general import ensure_dir
from src.utils.label_map import DISPLAY_NAMES
import pandas as pd
import os


class Evaluator:
    def __init__(self, num_classes: int, model_name: str,
                 results_dir: str, score_threshold: float = 0.01):

        self.num_classes      = num_classes
        self.model_name       = model_name
        self.results_dir      = results_dir
        self.score_threshold  = score_threshold

        ensure_dir(results_dir)

        self.metric = MeanAveragePrecision(
            iou_type="bbox",
            class_metrics=True
        )

    def reset(self):
        self.metric.reset()

    def update(self, predictions, targets):
        preds_cpu = []

        for p in predictions:
            scores = p["scores"]
            boxes  = p["boxes"]
            labels = p["labels"]

            keep   = scores >= self.score_threshold
            boxes  = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            if len(boxes) == 0:
                boxes  = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,),   dtype=torch.int64)
                scores = torch.zeros((0,),   dtype=torch.float32)

            preds_cpu.append({
                "boxes":  boxes.cpu(),
                "labels": labels.cpu(),
                "scores": scores.cpu()
            })

        targets_cpu = []

        for t in targets:
            boxes  = t["boxes"]
            labels = t["labels"]

            if len(boxes) == 0:
                boxes  = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,),   dtype=torch.int64)

            targets_cpu.append({
                "boxes":  boxes.cpu(),
                "labels": labels.cpu()
            })

        self.metric.update(preds_cpu, targets_cpu)

    def compute(self):
        return self.metric.compute()

    @torch.no_grad()
    def evaluate(self, model, dataloader, device, is_retinanet=False):
        model.eval()
        self.reset()

        print(f"\nEvaluating {self.model_name}")
        print(f"Score threshold: {self.score_threshold}\n")

        for images, targets in dataloader:
            images      = [img.to(device) for img in images]
            predictions = model(images)

            if is_retinanet:
                predictions = [
                    {**p, "labels": p["labels"] + 1}
                    for p in predictions
                ]

            self.update(predictions, targets)

        results = self.compute()
        self._print_results(results)
        self._save_results(results)

        return results

    def _print_results(self, results):
        print(f"\n{'='*60}")
        print(f"Results: {self.model_name}")
        print(f"{'='*60}")

        print(f"mAP @ IoU 0.50       : {results['map_50'].item():.4f}")
        print(f"mAP @ IoU 0.75       : {results['map_75'].item():.4f}")
        print(f"mAP @ IoU 0.50:0.95  : {results['map'].item():.4f}")

        print(f"\nMean Recall @1   : {results['mar_1'].item():.4f}")
        print(f"Mean Recall @10  : {results['mar_10'].item():.4f}")
        print(f"Mean Recall @100 : {results['mar_100'].item():.4f}")

        # FIX C (Phase D.3). This heading said "Per-Class AP @ IoU 0.50",
        # but TorchMetrics' map_per_class returns AP averaged over the COCO
        # IoU sweep 0.50:0.95, NOT AP@0.50. Verified against the archives:
        # for RetinaNeXt seed 42, Healthy prints 0.5594, which equals the
        # ap5095 column of per_class_test.csv, while its ap50 is 0.8138.
        # Only the label was wrong; the value is unchanged.
        print(f"\nPer-Class AP @ IoU 0.50:0.95  "
              f"(TorchMetrics map_per_class, COCO sweep - NOT AP@0.50)")
        print(f"{'-'*55}")

        per_class_ap  = results.get("map_per_class",     None)
        per_class_rec = results.get("mar_100_per_class", None)
        class_ids     = results.get("classes",           None)

        if per_class_ap is not None and len(per_class_ap) > 0:
            for i, ap in enumerate(per_class_ap):
                label      = int(class_ids[i].item()) if class_ids is not None else i + 1
                class_name = DISPLAY_NAMES.get(label, f"Class {label}")
                ap_val     = ap.item()
                rec_val    = per_class_rec[i].item() if per_class_rec is not None else -1

                if ap_val < 0:
                    print(f"[{label}] {class_name:<25} AP: N/A  Recall: N/A")
                else:
                    bar = "█" * int(ap_val * 30)
                    print(
                        f"[{label}] {class_name:<25} "
                        f"AP: {ap_val:.4f}  Recall: {rec_val:.4f} {bar}"
                    )
        else:
            print("Per-class AP not available")

        print(f"{'='*60}")

    def _save_results(self, results):
        summary = {
            "model":      self.model_name,
            "mAP_50":     round(results["map_50"].item(), 4),
            "mAP_75":     round(results["map_75"].item(), 4),
            "mAP_50_95":  round(results["map"].item(),   4),
            "mar_1":      round(results["mar_1"].item(),  4),
            "mar_10":     round(results["mar_10"].item(), 4),
            "mar_100":    round(results["mar_100"].item(),4),
        }

        per_class_ap  = results.get("map_per_class",     None)
        per_class_rec = results.get("mar_100_per_class", None)
        class_ids     = results.get("classes",           None)

        if per_class_ap is not None:
            for i, ap in enumerate(per_class_ap):
                label      = int(class_ids[i].item()) if class_ids is not None else i + 1
                class_name = DISPLAY_NAMES.get(label, f"class_{label}")
                safe_name  = class_name.replace(" ", "_")
                ap_val     = ap.item()
                rec_val    = per_class_rec[i].item() if per_class_rec is not None else -1

                # FIX C (Phase D.3). These values are AP@0.50:0.95, not
                # AP@0.50. The column is written under BOTH names:
                #   AP5095_<class>  the correct, unambiguous name
                #   AP_<class>      kept so existing downstream scripts and
                #                   every archived CSV stay readable
                # No value changes. For per-class AP@0.50 use the ap50
                # column of per_class_test.csv, which is what Figure 4 and
                # Table 3 read.
                v_ap  = round(ap_val,  4) if ap_val  >= 0 else None
                v_rec = round(rec_val, 4) if rec_val >= 0 else None
                summary[f"AP5095_{safe_name}"] = v_ap
                summary[f"AP_{safe_name}"]     = v_ap   # legacy alias
                summary[f"Recall_{safe_name}"] = v_rec

        df   = pd.DataFrame([summary])
        path = os.path.join(
            self.results_dir,
            f"{self.model_name}_eval_results.csv"
        )
        df.to_csv(path, index=False)
        print(f"\nResults saved to: {path}")
