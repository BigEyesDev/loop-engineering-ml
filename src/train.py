import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import torch
import mlflow
import mlflow.pytorch

import os

# Load .env from project root so API keys are available regardless of shell environment
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from data_loader import load_eurosat_splits, make_dataloaders, build_image_transforms, EUROSAT_CLASS_NAMES
from evaluate import evaluate_model, save_prediction_grid
from claude_advisor import request_config_update
from utils import load_config, is_plateau, setup_mlflow_experiment


def load_vit_model(num_classes: int, device: torch.device):
    """Load ViTForImageClassification from google/vit-base-patch16-224 with num_classes outputs; returns model."""
    from transformers import ViTForImageClassification
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model.to(device)


def load_efficientnet_model(num_classes: int, device: torch.device):
    """Load pretrained EfficientNetB0 via timm with num_classes outputs; returns model."""
    import timm
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    return model.to(device)


def apply_backbone_freeze(model, freeze: bool) -> None:
    """Freeze or unfreeze all ViT backbone layers, leaving the classifier head always trainable."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = not freeze


def build_optimizer(model, learning_rate: float, weight_decay: float):
    """Build AdamW over trainable parameters only; returns optimizer."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)


def train_one_epoch(model, dataloader, optimizer, criterion, device: torch.device) -> float:
    """Run one full training epoch with gradient updates; returns mean batch loss."""
    model.train()
    total_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="EuroSAT adaptive fine-tuning loop")
    parser.add_argument("--config", default="config/image_eurosat.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    base_config = load_config(config_path)
    smoke_test = args.smoke_test

    num_epochs = 2 if smoke_test else base_config["num_epochs"]
    device = torch.device("cpu") if smoke_test else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = base_config["num_classes"]

    print(f"Device: {device} | Epochs: {num_epochs} | Smoke test: {smoke_test}")

    train_split, val_split, test_split = load_eurosat_splits(smoke_test=smoke_test)

    if smoke_test:
        model = load_efficientnet_model(num_classes, device)
        run_name = "smoke_test_efficientnet_b0"
    else:
        model = load_vit_model(num_classes, device)
        run_name = "vit_base_eurosat"

    criterion = torch.nn.CrossEntropyLoss()
    setup_mlflow_experiment("loop_engineering_v1")

    active_batch_size = 4 if smoke_test else base_config["batch_size"]
    train_loader, val_loader, _ = make_dataloaders(
        train_split, val_split, test_split, batch_size=active_batch_size
    )

    current_config = load_config(config_path)
    active_freeze = current_config.get("freeze_backbone", False) and not smoke_test
    if active_freeze:
        apply_backbone_freeze(model, freeze=True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Backbone frozen — {trainable:,} trainable params (classifier head only)")

    optimizer = build_optimizer(model, current_config["learning_rate"], current_config["weight_decay"])

    f1_history = []
    epoch_history_for_claude = []
    claude_call_count = 0
    best_f1 = -1.0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": "efficientnet_b0" if smoke_test else base_config["model"],
            "dataset": base_config["dataset"],
        })

        for epoch in range(1, num_epochs + 1):
            current_config = load_config(config_path)
            new_freeze = current_config.get("freeze_backbone", False) and not smoke_test

            if new_freeze != active_freeze:
                active_freeze = new_freeze
                apply_backbone_freeze(model, active_freeze)
                optimizer = build_optimizer(model, current_config["learning_rate"], current_config["weight_decay"])
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  Backbone {'frozen' if active_freeze else 'unfrozen'} — optimizer rebuilt ({trainable:,} trainable params)")
            else:
                for pg in optimizer.param_groups:
                    pg["lr"] = current_config["learning_rate"]

            if not smoke_test:
                new_batch_size = current_config["batch_size"]
                if new_batch_size != active_batch_size:
                    active_batch_size = new_batch_size
                    train_loader, val_loader, _ = make_dataloaders(
                        train_split, val_split, test_split, batch_size=active_batch_size
                    )

            print(f"Epoch {epoch}/{num_epochs} | LR: {current_config['learning_rate']} | Batch: {active_batch_size}")

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            metrics = evaluate_model(model, val_loader, device)

            print(f"  train_loss={train_loss:.4f} | f1_macro={metrics['f1_macro']:.4f}")

            f1_history.append(metrics["f1_macro"])
            epoch_history_for_claude.append({
                "epoch": epoch,
                "f1": round(metrics["f1_macro"], 4),
                "lr": current_config["learning_rate"],
                "batch_size": active_batch_size,
                "freeze_backbone": active_freeze,
            })

            tokens_this_epoch = 0
            claude_suggested = False

            if smoke_test:
                plateau_detected = (epoch == 1)
            else:
                plateau_detected = is_plateau(
                    f1_history,
                    current_config["min_delta"],
                    current_config.get("plateau_window", 2),
                )

            if plateau_detected and claude_call_count < current_config["max_claude_calls"]:
                print(f"  Plateau detected — calling Claude advisor (call {claude_call_count + 1})")
                tokens_this_epoch = request_config_update(
                    current_config, epoch_history_for_claude, epoch, config_path
                )
                claude_call_count += 1
                claude_suggested = True
                print(f"  Claude responded ({tokens_this_epoch} tokens). Config updated.")

                current_config = load_config(config_path)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_config["learning_rate"]

            mlflow.log_metrics(
                {
                    "f1_macro": metrics["f1_macro"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "train_loss": train_loss,
                    "claude_tokens_used": float(tokens_this_epoch),
                    "learning_rate": current_config["learning_rate"],
                    "batch_size": float(active_batch_size),
                    "claude_suggested": float(claude_suggested),
                },
                step=epoch,
            )

            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]
                mlflow.pytorch.log_model(
                    model,
                    name="best_model",
                    serialization_format="pickle",
                )
                print(f"  New best F1: {best_f1:.4f} — model artifact saved")

            if not smoke_test and metrics["f1_macro"] >= base_config["target_f1"]:
                print(f"  Target F1 {base_config['target_f1']} reached. Stopping.")
                break

        grid_path = save_prediction_grid(
            model, val_split, build_image_transforms(), device, EUROSAT_CLASS_NAMES
        )
        mlflow.log_artifact(grid_path, artifact_path="visualizations")
        os.unlink(grid_path)

        run = mlflow.active_run()
        experiment_id = run.info.experiment_id
        run_id = run.info.run_id

    print(f"\nRun complete. Best F1: {best_f1:.4f} | Claude calls: {claude_call_count}")
    print(f"MLflow UI:  http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}")


if __name__ == "__main__":
    main()
