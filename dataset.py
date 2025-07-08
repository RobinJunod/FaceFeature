#%%
import torch
import numpy as np
import albumentations as A
import cv2

class FaceKeypointDataset(torch.utils.data.Dataset):
    """
    This class handles the loading and augmentation of the dataset.
    It uses Albumentations for image augmentations and PyTorch for data handling.
    """
    def __init__(self, df, augment=False):
        self.imgs  = df["Image"].values                                    # strings of 96*96 ints
        kp_px      = df.drop("Image", axis=1).values.astype("float32")     # (N,30) pixel coords
        self.mask  = ~np.isnan(kp_px)                                      # True where landmark exists
        kp_px[np.isnan(kp_px)] = 0.0

        self.kp_px = kp_px                                                 # **keep in pixel space**
        self.augment = augment
        self.transform = A.Compose(
        [
            # A.HorizontalFlip(p=0.5), # Removed because buggy
            # gentler geometry
            A.Affine(scale=(0.2, 0.8),      # 80 – 115 %
                    keep_ratio=True,
                    translate_percent=(-0.35, 0.35),
                    rotate=(-25, 25),
                    fit_output=False,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=np.random.uniform(0,1),  # fill black pixels with random values
                    p=0.85),
            
            A.CoarseDropout(fill="random", p=0.3),
            
            # photometric
            A.OneOf([
                A.RandomBrightnessContrast(.3, .3),
                A.RandomGamma((80,120)),
                A.CLAHE((2,4)),
            ], p=0.15),

            A.MotionBlur(3, p=0.05),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)   # <-- see step 3
    )
        # ----------------------------------------------------------------

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # ---------- image ----------
        img = np.fromstring(self.imgs[idx], sep=" ", dtype=np.uint8).reshape(96, 96)
        img = img.astype(np.float32) / 255.0                               # to [0,1]

        # ---------- key‑points ----------
        kp_px  = self.kp_px[idx].copy()                                    # (30,) pixel coords
        mask   = self.mask[idx].copy()

        # Albumentations wants a list of present key‑points
        keypoints, idx_map = [], []                                        # idx_map → original index
        for i in range(0, 30, 2):
            if mask[i]:
                keypoints.append((kp_px[i], kp_px[i+1]))
                idx_map.append(i)                                          # e.g. 0,2,4,…

        # ---------- apply augmentation ----------
        if self.augment:
            transformed = self.transform(image=img, keypoints=keypoints)
            img         = transformed['image']
            
            # fill black pixel with random values
            # img[img == 0] = np.random.uniform(0.0, 0.1, size=img[img == 0].shape)
            
            keypoints    = transformed['keypoints']

            # rebuild full (30,) vector
            kp_px[:] = 0.0
            mask[:]  = False
            for kpt, orig_i in zip(keypoints, idx_map):
                x, y = kpt
                # drop points that leave the frame
                if 0 <= x < 96 and 0 <= y < 96 and not np.isnan(x) and not np.isnan(y):
                    kp_px[orig_i]     = x
                    kp_px[orig_i + 1] = y
                    mask[orig_i] = mask[orig_i + 1] = True

        # ---------- normalise for the model ----------
        kp_norm = kp_px / 96.0                                             # [0,1]
        img     = torch.from_numpy(img).unsqueeze(0)                       # (1,96,96)
        kp_norm = torch.from_numpy(kp_norm.astype(np.float32))             # (30,)
        mask    = torch.from_numpy(mask)                                   # (30,) bool

        return img, kp_norm, mask
    


def show_img(img, kp, mask):
    import matplotlib.pyplot as plt
    img = img.squeeze().numpy()
    plt.imshow(img, cmap="gray")
    for i in range(0, len(kp), 2):
        if mask[i]:
            plt.scatter(kp[i], kp[i+1], c="r", s=10)
    plt.show()
#%%
if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader

    df = pd.read_csv("training.csv")
    dataset = FaceKeypointDataset(df, augment=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    for imgs, kps, masks in dataloader:
        print(imgs.shape, kps.shape, masks.shape)
        for i in range(len(imgs)):
            show_img(imgs[i], kps[i]*96, masks[i])
        break

# %%
