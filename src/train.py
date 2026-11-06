"""
Training Script for the RBM Recommendation Model
Trains the RBM on MovieLens 100K and saves the model.
"""

import os
import sys
import time
import json
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_preprocessing import prepare_data
from src.rbm_model import RBM

# ── Configuration ──────────────────────────────────────────────
CONFIG = {
    "n_hidden": 128,         # Hidden units (latent features)
    "k": 5,                  # CD-k steps
    "epochs": 50,            # Training epochs
    "batch_size": 64,        # Batch size
    "learning_rate": 0.01,   # Initial learning rate
    "momentum": 0.5,         # Momentum coefficient
    "weight_decay": 0.0001,  # L2 regularization
    "binary_threshold": 3,   # Rating binarization threshold
}


def train():
    """Train the RBM model and save it."""
    print("=" * 60)
    print("  RBM Movie Recommendation — Training")
    print("=" * 60)

    # ── Data Preparation ──
    print("\n[1/3] Preparing data...")
    data = prepare_data(
        test_size=0.2,
        binary=True,
        threshold=CONFIG["binary_threshold"]
    )

    train_matrix = torch.FloatTensor(data["train_matrix"])
    test_matrix = torch.FloatTensor(data["test_matrix"])

    n_users, n_movies = train_matrix.shape

    # Create masks (1 where user has rated)
    train_mask = (train_matrix != 0).float()
    test_mask = (test_matrix != 0).float()

    # Convert to 0/1 scale for binary RBM (from -1/0/1 to 0/0.5/1)
    train_input = (train_matrix + 1) / 2  # -1->0, 0->0.5, 1->1
    test_input = (test_matrix + 1) / 2

    # ── Model Initialization ──
    print(f"\n[2/3] Initializing RBM (visible={n_movies}, hidden={CONFIG['n_hidden']})...")
    rbm = RBM(
        n_visible=n_movies,
        n_hidden=CONFIG["n_hidden"],
        k=CONFIG["k"],
        learning_rate=CONFIG["learning_rate"],
        momentum=CONFIG["momentum"],
        weight_decay=CONFIG["weight_decay"],
    )

    total_params = sum(p.numel() for p in rbm.parameters())
    print(f"   Total parameters: {total_params:,}")

    # ── Training Loop ──
    print(f"\n[3/3] Training for {CONFIG['epochs']} epochs...")
    print("-" * 60)

    batch_size = CONFIG["batch_size"]
    n_batches = n_users // batch_size
    history = {"train_error": [], "test_error": []}

    best_test_error = float("inf")
    start_time = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        epoch_error = 0.0

        # Increase momentum after epoch 5
        if epoch == 6:
            rbm.momentum = 0.9

        # Shuffle users
        perm = torch.randperm(n_users)

        for batch_idx in range(n_batches):
            idx = perm[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            v_batch = train_input[idx]
            m_batch = train_mask[idx]

            error = rbm.contrastive_divergence(v_batch, m_batch)
            epoch_error += error

        avg_train_error = epoch_error / n_batches

        # Test error
        with torch.no_grad():
            pred = rbm.predict(test_input, test_mask)
            test_error = torch.sum(
                (test_input - pred) ** 2 * test_mask
            ) / torch.sum(test_mask)
            test_error = test_error.item()

        history["train_error"].append(avg_train_error)
        history["test_error"].append(test_error)

        # Progress bar
        bar_len = 30
        filled = int(bar_len * epoch / CONFIG["epochs"])
        bar = "#" * filled + "-" * (bar_len - filled)
        elapsed = time.time() - start_time

        print(
            f"  Epoch {epoch:3d}/{CONFIG['epochs']} [{bar}] "
            f"Train: {avg_train_error:.4f}  Test: {test_error:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # Save best model
        if test_error < best_test_error:
            best_test_error = test_error
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "models"
            )
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "rbm_trained.pt")
            torch.save({
                "model_state": rbm.state_dict(),
                "config": CONFIG,
                "n_visible": n_movies,
                "n_hidden": CONFIG["n_hidden"],
            }, model_path)

    # ── Summary ──
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"  Training complete in {total_time:.1f}s")
    print(f"  Best test error: {best_test_error:.4f}")
    print(f"  Model saved to: models/rbm_trained.pt")

    # Save training history
    history_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "training_history.json"
    )
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 60)
    return rbm, history


if __name__ == "__main__":
    train()
