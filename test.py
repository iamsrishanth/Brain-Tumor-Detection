"""
Brain Tumor MRI Classifier — Testing & Evaluation Script
=========================================================
Evaluates the trained model on the Testing directory.
Generates: confusion matrix, classification report, per-class accuracy,
sample predictions with confidence scores.

Usage:
    python test.py
    python test.py --model models/best_model.keras
"""

import os
import subprocess
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pull latest changes before running
_repo_dir = os.path.dirname(os.path.abspath(__file__))
print("Pulling latest changes from git...")
try:
    result = subprocess.run(
        ["git", "pull"], cwd=_repo_dir, capture_output=True, text=True, timeout=30
    )
    print(result.stdout.strip() if result.stdout.strip() else "Already up to date.")
    if result.stderr and "error" in result.stderr.lower():
        print(f"Git warning: {result.stderr.strip()}")
except Exception as e:
    print(f"Git pull skipped: {e}")
print()
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "Testing")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
IMG_SIZE = 300
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def parse_args():
    parser = argparse.ArgumentParser(description="Test brain tumor classifier")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(MODEL_DIR, "best_model.keras"),
        help="Path to trained model",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    return parser.parse_args()


def load_model_and_labels(model_path):
    """Load model and class labels."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun train.py first.")

    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded: {model_path}")

    labels_path = os.path.join(MODEL_DIR, "class_labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            classes = json.load(f)
    else:
        classes = CLASSES
        print("Warning: class_labels.json not found, using default class order")

    return model, classes


def create_test_generator(batch_size):
    """Create test data generator."""
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,  # Important: keep order for evaluation
    )
    return test_gen


def plot_confusion_matrix(cm, classes, save_path):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix — Brain Tumor Classification", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def plot_per_class_accuracy(cm, classes, save_path):
    """Plot per-class accuracy bar chart."""
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, per_class_acc, color=["#2196F3", "#FF9800", "#4CAF50", "#E91E63"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Per-Class Accuracy", fontsize=15)

    for bar, acc in zip(bars, per_class_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.1%}",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class accuracy chart saved: {save_path}")


def plot_confidence_distribution(all_confidences, all_preds, all_true, classes, save_path):
    """Plot confidence score distributions per class."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, cls in enumerate(classes):
        # Confidence scores where this class was predicted
        mask = all_preds == i
        if mask.sum() == 0:
            continue
        confs = all_confidences[mask, i]
        correct = all_true[mask] == i

        axes[i].hist(confs[correct], bins=20, alpha=0.7, label="Correct", color="#4CAF50")
        axes[i].hist(confs[~correct], bins=20, alpha=0.7, label="Wrong", color="#F44336")
        axes[i].set_title(f"{cls}", fontsize=13)
        axes[i].set_xlabel("Confidence")
        axes[i].set_ylabel("Count")
        axes[i].legend()

    plt.suptitle("Prediction Confidence Distribution", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confidence distribution saved: {save_path}")


def show_sample_predictions(model, test_gen, classes, num_samples=16):
    """Display sample predictions with images."""
    test_gen.reset()
    batch_x, batch_y = next(test_gen)
    predictions = model.predict(batch_x, verbose=0)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    for i in range(min(num_samples, len(batch_x))):
        img = batch_x[i]
        true_label = classes[np.argmax(batch_y[i])]
        pred_label = classes[np.argmax(predictions[i])]
        confidence = np.max(predictions[i]) * 100

        axes[i].imshow(img)
        color = "#4CAF50" if true_label == pred_label else "#F44336"
        axes[i].set_title(
            f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)",
            fontsize=10,
            color=color,
            fontweight="bold",
        )
        axes[i].axis("off")

    plt.suptitle("Sample Predictions", fontsize=16, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "sample_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample predictions saved: {save_path}")


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ─── Load model ───────────────────────────────────────────────────────────
    model, classes = load_model_and_labels(args.model)

    # ─── Create test generator ────────────────────────────────────────────────
    test_gen = create_test_generator(args.batch_size)

    # ─── Run predictions ──────────────────────────────────────────────────────
    print("\nRunning predictions on test set...")
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    # ─── Metrics ──────────────────────────────────────────────────────────────
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(classes))
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 55)
    for i, cls in enumerate(classes):
        print(
            f"{cls:<15} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10d}"
        )

    # Macro averages
    print("-" * 55)
    print(
        f"{'Macro Avg':<15} {precision.mean():>10.4f} {recall.mean():>10.4f} {f1.mean():>10.4f} {support.sum():>10d}"
    )

    # Full classification report
    print("\n\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    # ─── Confusion Matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    plot_confusion_matrix(cm, classes, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_per_class_accuracy(cm, classes, os.path.join(RESULTS_DIR, "per_class_accuracy.png"))
    plot_confidence_distribution(
        predictions, y_pred, y_true, classes,
        os.path.join(RESULTS_DIR, "confidence_distribution.png"),
    )

    # ─── Sample predictions ───────────────────────────────────────────────────
    show_sample_predictions(model, test_gen, classes)

    # ─── Save results summary ─────────────────────────────────────────────────
    results_summary = {
        "model": args.model,
        "overall_accuracy": float(accuracy),
        "per_class": {
            cls: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }
            for i, cls in enumerate(classes)
        },
        "macro_avg": {
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1_score": float(f1.mean()),
        },
    }

    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
