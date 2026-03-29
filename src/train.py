# train.py

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from utils import SegmentationDataset
from config import *
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = SegmentationDataset( OUTPUT_DIR+"patches/images",   OUTPUT_DIR+"patches/masks")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=len(CLASS_MAP)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()
#loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    total_train_acc = 0
    total_loss = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # Calculate and accumulate accuracy (example for binary segmentation)
        #outputs = (preds > 0.0).float()
        #accuracy = (outputs == masks).float().mean().item()
        #total_train_acc += accuracy
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Current Time at Epoch End : ",datetime.now().time())

torch.save(model.state_dict(), OUTPUT_DIR+"models/unet_with_45_Epochs.pth")
# plot the training loss and accuracy
'''
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
'''