import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_model(model, dataloader, device: torch.device) -> dict:
    """Run inference on dataloader and return dict with f1_macro, precision_macro, recall_macro."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            predictions = logits.argmax(dim=-1).cpu().numpy()
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.numpy().tolist())

    return {
        "f1_macro": f1_score(all_labels, all_predictions, average="macro", zero_division=0),
        "precision_macro": precision_score(all_labels, all_predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(all_labels, all_predictions, average="macro", zero_division=0),
    }


def save_prediction_grid(model, val_split, transform, device: torch.device, class_names: list, num_samples: int = 6) -> str:
    """Run inference on num_samples validation images and save a 6x3 grid (image/prediction/gt); returns artifact path."""
    model.eval()
    n = min(num_samples, len(val_split))
    samples = [val_split[i] for i in range(n)]
    images_pil = [s["image"].convert("RGB") for s in samples]
    labels = [s["label"] for s in samples]

    tensors = torch.stack([transform(img) for img in images_pil]).to(device)
    with torch.no_grad():
        outputs = model(tensors)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()

    fig, axes = plt.subplots(n, 3, figsize=(9, 2.8 * n))
    fig.patch.set_facecolor("white")

    for j, title in enumerate(["Image", "Prediction", "Ground Truth"]):
        axes[0, j].set_title(title, fontweight="bold", fontsize=11, pad=8)

    for i in range(n):
        correct = preds[i] == labels[i]

        axes[i, 0].imshow(np.array(images_pil[i]))
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        for spine in axes[i, 0].spines.values():
            spine.set_visible(False)

        for col, (text, color) in enumerate(
            [
                (class_names[preds[i]], "#27ae60" if correct else "#e74c3c"),
                (class_names[labels[i]], "#2980b9"),
            ],
            start=1,
        ):
            ax = axes[i, col]
            ax.set_facecolor(color)
            ax.text(
                0.5, 0.5, text,
                ha="center", va="center",
                fontsize=10, fontweight="bold", color="white",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout(pad=1.5)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return tmp.name
