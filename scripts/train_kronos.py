"""Resumable Kronos fine-tuning script.

Wraps the Kronos finetune_csv pipeline with checkpoint-based resume support.
You can train N epochs at a time, stop, and continue later from where you left off.

Usage:
    # Train tokenizer for 5 epochs
    python scripts/train_kronos.py --config configs/config_crypto_5m.yaml --phase tokenizer --epochs 5

    # Resume tokenizer training for 5 more epochs
    python scripts/train_kronos.py --config configs/config_crypto_5m.yaml --phase tokenizer --epochs 5

    # Train predictor for 3 epochs
    python scripts/train_kronos.py --config configs/config_crypto_5m.yaml --phase predictor --epochs 3

    # Check current training status
    python scripts/train_kronos.py --config configs/config_crypto_5m.yaml --status
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add kronos to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KRONOS_PATH = os.path.join(PROJECT_ROOT, "third_party", "kronos")
FINETUNE_PATH = os.path.join(KRONOS_PATH, "finetune_csv")
sys.path.insert(0, KRONOS_PATH)
sys.path.insert(0, FINETUNE_PATH)

from model import Kronos, KronosTokenizer
from config_loader import CustomFinetuneConfig
from finetune_base_model import CustomKlineDataset


TOP_K = 5  # number of best models to keep


def get_checkpoint_path(save_dir: str, phase: str) -> str:
    return os.path.join(save_dir, f"{phase}_checkpoint.pt")


def save_top_k_model(save_dir: str, model, epoch: int, val_loss: float):
    """Save model if it's among the top-K best, remove worst if over limit."""
    import json
    import shutil

    ranking_path = os.path.join(save_dir, "ranking.json")

    # Load existing ranking
    if os.path.exists(ranking_path):
        with open(ranking_path) as f:
            ranking = json.load(f)
    else:
        ranking = []

    # Check if this model deserves a spot
    if len(ranking) >= TOP_K and val_loss >= ranking[-1]["val_loss"]:
        return False  # not good enough

    # Save this model
    model_dir = os.path.join(save_dir, f"epoch_{epoch}_loss_{val_loss:.6f}")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)

    # Add to ranking
    ranking.append({"epoch": epoch, "val_loss": val_loss, "path": model_dir})
    ranking.sort(key=lambda x: x["val_loss"])

    # Remove worst if over limit
    while len(ranking) > TOP_K:
        worst = ranking.pop()
        if os.path.exists(worst["path"]):
            shutil.rmtree(worst["path"])
            print(f"  Removed model: {os.path.basename(worst['path'])}")

    # Always keep best_model/ as a copy of rank 1
    best_path = os.path.join(save_dir, "best_model")
    if ranking[0]["path"] != best_path:
        if os.path.exists(best_path):
            shutil.rmtree(best_path)
        shutil.copytree(ranking[0]["path"], best_path)

    # Save updated ranking
    with open(ranking_path, "w") as f:
        json.dump(ranking, f, indent=2)

    rank = next(i for i, r in enumerate(ranking) if r["epoch"] == epoch) + 1
    print(f"  Saved model: epoch {epoch}, val_loss={val_loss:.6f} (rank {rank}/{len(ranking)})")
    return True


def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int,
                    best_val_loss: float, total_epochs_done: int):
    """Save full training state for resume."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "total_epochs_done": total_epochs_done,
    }, path)
    print(f"  Checkpoint saved: epoch {total_epochs_done}, val_loss={best_val_loss:.6f}")


def load_checkpoint(path: str, model, optimizer, scheduler, device):
    """Restore training state from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["best_val_loss"], ckpt["total_epochs_done"]


def create_dataloaders(config, data_type="train"):
    """Create train and val dataloaders from config."""
    train_dataset = CustomKlineDataset(
        data_path=config.data_path,
        data_type="train",
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    val_dataset = CustomKlineDataset(
        data_path=config.data_path,
        data_type="val",
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed + 1,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader, train_dataset, val_dataset


def train_tokenizer_epochs(config, device, num_epochs: int):
    """Train tokenizer for num_epochs, resuming from checkpoint if available."""
    save_dir = config.tokenizer_save_path
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = get_checkpoint_path(config.base_save_path, "tokenizer")

    # Load pre-trained tokenizer
    print("Loading pre-trained tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    tokenizer = tokenizer.to(device)
    print(f"Tokenizer parameters: {sum(p.numel() for p in tokenizer.parameters()):,}")

    # Create dataloaders
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(config)

    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=config.tokenizer_learning_rate,
        weight_decay=config.adam_weight_decay,
    )
    # CosineAnnealingWarmRestarts: LR resets to max at the start of each epoch
    # T_0 = steps per epoch, so LR cycles once per epoch
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=steps_per_epoch, T_mult=1, eta_min=1e-6,
    )

    best_val_loss = float("inf")
    start_epoch = 0
    total_epochs_done = 0

    # Resume from checkpoint if exists
    if os.path.exists(ckpt_path):
        print(f"\nResuming from checkpoint: {ckpt_path}")
        start_epoch, best_val_loss, total_epochs_done = load_checkpoint(
            ckpt_path, tokenizer, optimizer, scheduler, device,
        )
        print(f"  Resuming after epoch {total_epochs_done}, best_val_loss={best_val_loss:.6f}")
        # Reset LR and rebuild scheduler — warm restarts handles per-epoch cycling
        for pg in optimizer.param_groups:
            pg["lr"] = config.tokenizer_learning_rate
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=steps_per_epoch, T_mult=1, eta_min=1e-6,
        )
    else:
        print("\nStarting fresh tokenizer training")

    print(f"Will train for {num_epochs} epochs (epochs {total_epochs_done + 1} → {total_epochs_done + num_epochs})\n")

    accumulation_steps = getattr(config, "accumulation_steps", 1)

    for epoch_i in range(num_epochs):
        epoch_num = total_epochs_done + epoch_i + 1
        epoch_start = time.time()
        tokenizer.train()
        train_dataset.set_epoch_seed(epoch_num * 10000)

        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (ori_batch_x, _) in enumerate(train_loader):
            ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)

            total_loss = 0.0
            for j in range(accumulation_steps):
                bs = ori_batch_x.shape[0] // accumulation_steps
                batch_x = ori_batch_x[j * bs : (j + 1) * bs]

                zs, bsq_loss, _, _ = tokenizer(batch_x)
                z_pre, z = zs
                recon_loss = F.mse_loss(z_pre, batch_x) + F.mse_loss(z, batch_x)
                loss = (recon_loss + bsq_loss) / 2
                (loss / accumulation_steps).backward()
                total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += total_loss / accumulation_steps
            n_batches += 1

            if (batch_idx + 1) % config.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                avg = total_loss / accumulation_steps
                print(f"  [Epoch {epoch_num}, Step {batch_idx + 1}/{len(train_loader)}] "
                      f"LR: {lr:.2e}, Loss: {avg:.4f}")

        # Validation
        tokenizer.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)
                zs, _, _, _ = tokenizer(ori_batch_x)
                _, z = zs
                val_loss_sum += F.mse_loss(z, ori_batch_x).item() * ori_batch_x.size(0)
                val_count += ori_batch_x.size(0)

        avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0
        avg_val_loss = val_loss_sum / val_count if val_count > 0 else 0
        epoch_time = time.time() - epoch_start

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        improved = " ★ new best" if avg_val_loss <= best_val_loss else ""
        print(f"\n  Epoch {epoch_num} — train_loss: {avg_train_loss:.4f}, "
              f"val_loss: {avg_val_loss:.4f}, time: {epoch_time:.0f}s{improved}\n")

        # Save top-K models
        save_top_k_model(save_dir, tokenizer, epoch_num, avg_val_loss)

        # Save checkpoint after every epoch
        save_checkpoint(
            ckpt_path, tokenizer, optimizer, scheduler,
            epoch_i, best_val_loss, total_epochs_done + epoch_i + 1,
        )

    print(f"\nTokenizer training done. Total epochs: {total_epochs_done + num_epochs}, "
          f"best_val_loss: {best_val_loss:.6f}")
    print(f"Best model: {os.path.join(save_dir, 'best_model')}")
    print(f"Checkpoint: {ckpt_path}")


def train_predictor_epochs(config, device, num_epochs: int):
    """Train predictor for num_epochs, resuming from checkpoint if available."""
    save_dir = config.basemodel_save_path
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = get_checkpoint_path(config.base_save_path, "predictor")

    # Load fine-tuned tokenizer (required)
    tokenizer_path = config.finetuned_tokenizer_path
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Fine-tuned tokenizer not found at {tokenizer_path}")
        print("Train the tokenizer first: --phase tokenizer")
        sys.exit(1)

    print("Loading fine-tuned tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer = tokenizer.to(device)
    tokenizer.eval()

    # Load pre-trained predictor
    print("Loading pre-trained predictor...")
    model = Kronos.from_pretrained(config.pretrained_predictor_path)
    model = model.to(device)
    print(f"Predictor parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataloaders
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.predictor_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
    )
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=steps_per_epoch, T_mult=1, eta_min=1e-7,
    )

    best_val_loss = float("inf")
    total_epochs_done = 0

    if os.path.exists(ckpt_path):
        print(f"\nResuming from checkpoint: {ckpt_path}")
        _, best_val_loss, total_epochs_done = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, device,
        )
        print(f"  Resuming after epoch {total_epochs_done}, best_val_loss={best_val_loss:.6f}")
        for pg in optimizer.param_groups:
            pg["lr"] = config.predictor_learning_rate
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=steps_per_epoch, T_mult=1, eta_min=1e-7,
        )
    else:
        print("\nStarting fresh predictor training")

    print(f"Will train for {num_epochs} epochs (epochs {total_epochs_done + 1} → {total_epochs_done + num_epochs})\n")

    for epoch_i in range(num_epochs):
        epoch_num = total_epochs_done + epoch_i + 1
        epoch_start = time.time()
        model.train()
        train_dataset.set_epoch_seed(epoch_num * 10000)

        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, s1_loss, s2_loss = model.head.compute_loss(
                logits[0], logits[1], token_out[0], token_out[1],
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % config.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  [Epoch {epoch_num}, Step {batch_idx + 1}/{len(train_loader)}] "
                      f"LR: {lr:.2e}, Loss: {loss.item():.4f} (s1: {s1_loss.item():.4f}, s2: {s2_loss.item():.4f})")

        # Save checkpoint immediately after training (before validation)
        # so we don't lose the epoch if validation crashes
        avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0
        save_checkpoint(
            ckpt_path, model, optimizer, scheduler,
            epoch_i, best_val_loss, total_epochs_done + epoch_i + 1,
        )
        print(f"\n  Checkpoint saved. Running validation...")

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, _, _ = model.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1],
                )
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0
        epoch_time = time.time() - epoch_start

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        improved = " ★ new best" if avg_val_loss <= best_val_loss else ""
        print(f"\n  Epoch {epoch_num} — train_loss: {avg_train_loss:.4f}, "
              f"val_loss: {avg_val_loss:.4f}, time: {epoch_time:.0f}s{improved}\n")

        # Save top-K models (now with val_loss for ranking)
        save_top_k_model(save_dir, model, epoch_num, avg_val_loss)

        # Update checkpoint with val_loss
        save_checkpoint(
            ckpt_path, model, optimizer, scheduler,
            epoch_i, best_val_loss, total_epochs_done + epoch_i + 1,
        )

    print(f"\nPredictor training done. Total epochs: {total_epochs_done + num_epochs}, "
          f"best_val_loss: {best_val_loss:.6f}")
    print(f"Best model: {os.path.join(save_dir, 'best_model')}")
    print(f"Checkpoint: {ckpt_path}")


def show_status(config):
    """Show current training status for both phases."""
    print("=" * 60)
    print("Kronos Fine-Tuning Status")
    print("=" * 60)

    for phase in ["tokenizer", "predictor"]:
        ckpt_path = get_checkpoint_path(config.base_save_path, phase)
        if phase == "tokenizer":
            best_path = os.path.join(config.tokenizer_save_path, "best_model")
            target_epochs = config.tokenizer_epochs
        else:
            best_path = os.path.join(config.basemodel_save_path, "best_model")
            target_epochs = config.basemodel_epochs

        print(f"\n  {phase.upper()}")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            done = ckpt["total_epochs_done"]
            val_loss = ckpt["best_val_loss"]
            print(f"    Epochs completed: {done} / {target_epochs}")
            print(f"    Best val loss:    {val_loss:.6f}")
            print(f"    Checkpoint:       {ckpt_path}")
        else:
            print(f"    Not started")

        if os.path.exists(best_path):
            print(f"    Best model:       {best_path}")
        else:
            print(f"    Best model:       not saved yet")

        # Show top-K ranking
        save_dir = config.tokenizer_save_path if phase == "tokenizer" else config.basemodel_save_path
        ranking_path = os.path.join(save_dir, "ranking.json")
        if os.path.exists(ranking_path):
            import json
            with open(ranking_path) as f:
                ranking = json.load(f)
            print(f"    Top {len(ranking)} models:")
            for i, r in enumerate(ranking):
                print(f"      #{i+1}  epoch {r['epoch']:>3}  val_loss={r['val_loss']:.6f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Resumable Kronos fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--phase", type=str, choices=["tokenizer", "predictor"],
                        help="Which phase to train")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train in this run (default: 5)")
    parser.add_argument("--status", action="store_true",
                        help="Show current training status and exit")

    args = parser.parse_args()

    # Resolve config path relative to finetune_csv dir if not absolute
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(FINETUNE_PATH, config_path)
    config = CustomFinetuneConfig(config_path)

    if args.status:
        show_status(config)
        return

    if not args.phase:
        parser.error("--phase is required when not using --status")

    # Setup device
    if config.use_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.device_id}")
        gpu_name = torch.cuda.get_device_name(config.device_id)
        vram = torch.cuda.get_device_properties(config.device_id).total_memory / 1e9
        print(f"Using GPU: {gpu_name} ({vram:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("Using CPU (this will be slow)")

    os.makedirs(config.base_save_path, exist_ok=True)

    if args.phase == "tokenizer":
        train_tokenizer_epochs(config, device, args.epochs)
    elif args.phase == "predictor":
        train_predictor_epochs(config, device, args.epochs)


if __name__ == "__main__":
    main()
