

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import FaceKeypointDataset
from model import KeypointNet, KeypointNetM, ViTFaceKeypoint

############################################################
# Utility functions
############################################################

def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments as specified in README."""

    parser = argparse.ArgumentParser(
        description="Train a face key‑point detector with optional CNN/VIT back‑ends.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core hyper‑parameters (match README)
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini‑batch size")
    parser.add_argument(
        "--data-dir", type=str, default="", help="Folder containing training.csv"
    )
    parser.add_argument(
        "--out-dir", type=str, default="checkpoints", help="Directory for weights & TensorBoard logs"
    )

    # Extras — optional but handy
    parser.add_argument(
        "--model",
        type=str,
        default="cnn_adv",
        choices=["cnn", "vit", "cnn_adv"],
        help="Back‑end architecture to use",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint to resume training from"
    )
    parser.add_argument(
        "--save-every", type=int, default=20, help="Save a checkpoint every N epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Initial learning rate"
    )

    return parser.parse_args()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean‑squared‑error that ignores missing key‑points via `mask`."""
    se = (pred - target) ** 2 * mask
    denom = mask.sum(1).clamp_min(1.0)
    return (se.sum(1) / denom).mean()


def masked_huberloss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float = 1.0 / 9.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Masked Smooth‑L1 (Huber) loss that works with ViT key‑point models."""
    mask = mask.float()
    diff = torch.abs(pred - target)

    loss = torch.where(diff < delta, 0.5 * diff ** 2 / delta, diff - 0.5 * delta) * mask

    if reduction == "mean":
        denom = mask.sum(1).clamp_min(1.0)
        return (loss.sum(1) / denom).mean()
    if reduction == "sum":
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss  # "none"

############################################################
# Main training loop
############################################################

def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------------------
    # I/O setup
    # ---------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    csv_path = data_dir / "training.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Could not find {csv_path} — check --data-dir argument.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = out_dir / "tb_logs"
    writer = SummaryWriter(log_dir=str(tb_dir))

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_ds = FaceKeypointDataset(train_df, augment=True)
    val_ds = FaceKeypointDataset(val_df, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "cnn":
        model = KeypointNet().to(device)
    elif args.model == "vit":
        model = ViTFaceKeypoint().to(device)
    else:  # cnn_adv
        model = KeypointNetM().to(device)

    if args.resume is not None and Path(args.resume).is_file():
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from checkpoint {args.resume}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        # -------- train --------
        model.train()
        train_loss_accum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")

        for imgs, kps, msk in pbar:
            imgs, kps, msk = imgs.to(device), kps.to(device), msk.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(imgs)
            loss = masked_mse(preds, kps, msk)
            # loss = masked_huberloss(preds, kps, msk)  # uncomment for Huber loss
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        train_loss = train_loss_accum / len(train_loader)
        print(f"Epoch {epoch}: train loss = {train_loss:.4f}")

        # -------- validate --------
        model.eval()
        val_loss_accum = 0.0
        with torch.inference_mode():
            for imgs, kps, msk in val_loader:
                imgs, kps, msk = imgs.to(device), kps.to(device), msk.to(device)
                val_loss_accum += masked_mse(model(imgs), kps, msk).item()
                # val_loss_accum += masked_huberloss(preds, kps, msk)  # uncomment for Huber loss
        val_loss = val_loss_accum / len(val_loader)
        print(f"            val loss = {val_loss:.4f}")

        # -------- log to TensorBoard --------
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # -------- checkpoint --------
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = out_dir / f"{model.__class__.__name__}_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint ➜ {ckpt_path.relative_to(Path.cwd())}")

    writer.close()
    print("Training complete. TensorBoard logs written to", tb_dir)


if __name__ == "__main__":  # only executed when run as a script
    main()
    
    