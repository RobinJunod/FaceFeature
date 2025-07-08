import os, csv
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

from dataset import FaceKeypointDataset
from model import KeypointNet, ViTFaceKeypoint, KeypointNetM



def masked_mse(pred, target, mask):
    se = (pred - target) ** 2 * mask
    denom = mask.sum(1).clamp_min(1.0)
    return (se.sum(1) / denom).mean()

def masked_huberloss(pred, target, mask, delta=1.0/9.0, reduction="mean"):
    """
    Masked Huber (Smooth-L1) loss compatible with the ViT keypoint model.
    pred/target in [0, 1], mask is 1.0 for valid points.
    """
    mask = mask.float()           # make sure broadcast works on GPU
    diff = torch.abs(pred - target)

    loss = torch.where(
        diff < delta,
        0.5 * diff ** 2 / delta,  # quadratic region
        diff - 0.5 * delta,       # linear region
    ) * mask

    if reduction == "mean":
        # per-sample normalisation like the original masked loss
        denom = mask.sum(1).clamp_min(1.0)
        return (loss.sum(1) / denom).mean()
    elif reduction == "sum":
        return loss.sum() / mask.sum().clamp_min(1.0)
    else:                          # "none"
        return loss



CSV_PATH = "/kaggle/input/facialfeat-extracted/training.csv"  # adjust if needed
CKPT_IN = "/kaggle/input/keyppointnetm/pytorch/default/1/KeypointNetM_epoch300.pth" # if you want to resume training
OUTPUT_DIR = "/kaggle/working"  # where to save checkpoints

EPOCHS    = 300
SAVE_EVERY = 20
MODE='cnn_adv' # 'cnn', 'vit', 'cnn_adv' (KeypointNetM)

if __name__ == "__main__":
    # -------- load data --------
    df = pd.read_csv(CSV_PATH)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_ds = FaceKeypointDataset(train_df, augment=True)
    val_ds   = FaceKeypointDataset(val_df,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                            num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=2, pin_memory=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if MODE == 'cnn':
        model = KeypointNet().to(device)
    elif MODE == 'vit':
        model = ViTFaceKeypoint().to(device)
    elif MODE == 'cnn_adv':
        model = KeypointNetM().to(device)
    model_name = model.__class__.__name__
    
    # optional: resume from a checkpoint in /kaggle/input/
    if os.path.exists(CKPT_IN):
        model.load_state_dict(torch.load(CKPT_IN, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler    = GradScaler()          # AMP


    for epoch in range(1, EPOCHS + 1):
        # -------- training --------
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for imgs, kps, msk in pbar:
            imgs, kps, msk = imgs.to(device), kps.to(device), msk.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(imgs)
            loss  = masked_mse(preds, kps, msk)
            #loss =  masked_huberloss(preds, kps, msk)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        print(f"Epoch {epoch}: train loss = {train_loss:.4f}")

        # -------- validation --------
        model.eval()
        val_loss = 0.0
        with torch.inference_mode(), autocast():
            for imgs, kps, msk in val_loader:
                imgs, kps, msk = imgs.to(device), kps.to(device), msk.to(device)
                #val_loss += masked_huberloss(model(imgs), kps, msk).item()
                val_loss += masked_mse(model(imgs), kps, msk).item()
        val_loss /= len(val_loader)
        print(f"           val loss   = {val_loss:.4f}")

        # -------- save checkpoint --------
        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            
            outpath = os.path.join(OUTPUT_DIR, f"{model_name}_epoch{epoch}.pth")
            
            torch.save(model.state_dict(),outpath) # Change to your desired path
            print(f"Saved checkpoint for epoch {epoch}.")