"""
Brain Tumor MRI Classifier — Training Script
=============================================
Model: EfficientNetB3 (transfer learning + fine-tuning)
Classes: glioma, meningioma, notumor, pituitary

Usage:
    python train.py
    python train.py --epochs 30 --fine_tune_epochs 20 --batch_size 16
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
TEST_DIR = os.path.join(BASE_DIR, "Testing")
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_SIZE = 300  # EfficientNetB3 optimal input size
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Train brain tumor classifier")
    parser.add_argument("--epochs", type=int, default=25, help="Phase 1 epochs (frozen base)")
    parser.add_argument("--fine_tune_epochs", type=int, default=15, help="Phase 2 epochs (fine-tuning)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5, help="Fine-tuning learning rate")
    return parser.parse_args()


def create_data_generators(batch_size):
    """Create train/val/test generators with augmentation."""

    # Training augmentation — aggressive but MRI-appropriate
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest",
        validation_split=0.15,  # 15% for validation
    )

    # Test/val — just rescale
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        classes=CLASSES,
        shuffle=True,
        seed=42,
    )

    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        classes=CLASSES,
        shuffle=False,
        seed=42,
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )

    print(f"\nClass indices: {train_gen.class_indices}")
    print(f"Training samples:   {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Testing samples:    {test_gen.samples}\n")

    return train_gen, val_gen, test_gen


def build_model(num_classes):
    """Build EfficientNetB3 with custom classification head."""

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Load pre-trained EfficientNetB3 (ImageNet weights, no top)
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )

    # Freeze base model for Phase 1
    base_model.trainable = False

    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = BatchNormalization(name="bn_1")(x)
    x = Dense(256, activation="relu", name="dense_1")(x)
    x = Dropout(0.4, name="dropout_1")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = Dense(128, activation="relu", name="dense_2")(x)
    x = Dropout(0.3, name="dropout_2")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="brain_tumor_efficientnetb3")
    return model, base_model


def get_callbacks(model_dir):
    """Create training callbacks."""
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    return [
        ModelCheckpoint(
            os.path.join(model_dir, "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def compute_class_weights(train_gen):
    """Handle slight class imbalance (no: 1498 vs others: 1400)."""
    from sklearn.utils.class_weight import compute_class_weight

    unique_classes = np.unique(train_gen.classes)
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=train_gen.classes,
    )
    # Ensure all NUM_CLASSES are covered (default weight 1.0 for any missing)
    weights = {i: 1.0 for i in range(NUM_CLASSES)}
    for cls, w in zip(unique_classes, class_weights_arr):
        weights[int(cls)] = float(w)
    return weights


def main():
    args = parse_args()

    # ─── GPU check ────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU detected: {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU detected — training on CPU (will be slower)")

    # ─── Data ─────────────────────────────────────────────────────────────────
    train_gen, val_gen, test_gen = create_data_generators(args.batch_size)
    class_weights = compute_class_weights(train_gen)
    print(f"Class weights: {class_weights}\n")

    # ─── Build model ──────────────────────────────────────────────────────────
    model, base_model = build_model(NUM_CLASSES)
    model.summary(print_fn=lambda x: None)  # suppress verbose summary

    total_params = model.count_params()
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Non-trainable:    {total_params - trainable:,}\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Train classification head (base frozen)
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 1: Training classification head (base frozen)")
    print("=" * 60)

    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_phase1 = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=get_callbacks(MODEL_DIR),
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Fine-tune top layers of base model
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning top layers of EfficientNetB3")
    print("=" * 60)

    # Unfreeze the last 30 layers of the base model
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"Trainable params after unfreeze: {trainable:,}")

    model.compile(
        optimizer=Adam(learning_rate=args.fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_phase2 = model.fit(
        train_gen,
        epochs=args.fine_tune_epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=get_callbacks(MODEL_DIR),
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Final evaluation on test set
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    # Load best model
    best_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))

    test_loss, test_acc = best_model.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    # Save final model
    best_model.save(os.path.join(MODEL_DIR, "brain_tumor_classifier.keras"))
    print(f"\nModel saved to: {os.path.join(MODEL_DIR, 'brain_tumor_classifier.keras')}")

    # Save class labels
    import json

    with open(os.path.join(MODEL_DIR, "class_labels.json"), "w") as f:
        json.dump(CLASSES, f)
    print(f"Class labels saved to: {os.path.join(MODEL_DIR, 'class_labels.json')}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
