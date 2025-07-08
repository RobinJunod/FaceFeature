#%%
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


from  model import  KeypointNet, ViTFaceKeypoint, KeypointNet3, KeypointNetM
from dataset import FaceKeypointDataset


def show_img_pred(img, label, mask, pred):
    img = img.squeeze().numpy()
    plt.imshow(img, cmap="gray")
    for i in range(0, len(label), 2):
        if mask[i]:
            plt.scatter(label[i], label[i+1], c="r", s=10)
        plt.scatter(pred[i], pred[i+1], c="b", s=10)
    plt.show()
    

df = pd.read_csv('training.csv')
# Split the dataset
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
# Create dataset instances
val_dataset = FaceKeypointDataset(df_val, augment=True)
# Create DataLoaders
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = KeypointNetM().to(device)
model.load_state_dict(torch.load('weights\\KeypointNetM_epoch300.pth', map_location=device))

val_loss = 0.0
with torch.no_grad():
    i = 0
    for images, labels, masks in val_loader:
        i += 1
        images, labels, masks = images.to(device), labels.to(device), masks.to(device)
        # Show images and keypoints and predictions
        prediction = model(images)
        show_img_pred(images[0], labels[0]*96, masks[0], prediction[0]*96)
        if i > 5:
            break


# %%
