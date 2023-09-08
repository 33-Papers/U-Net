import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from unet import UNET
from torchmetrics import JaccardIndex

from utils import load_checkpoint, save_checkpoint, get_loaders

from config import (LEARNING_RATE, DEVICE, BATCH_SIZE, NUM_WORKERS, NUM_EPOCHS, IMAGE_HEIGHT, IMAGE_WIDTH, PIN_MEMORY,
                    LOAD_MODEL, TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR)


def train(train_loader, val_loader, device, optimizer, model, criteria, iou_score):
    train_losses, val_losses, iou = [], [], []
    for e in tqdm(range(NUM_EPOCHS)):
        model.train()

        running_train_loss, running_val_loss, running_iou = 0, 0, 0

        for i, data in enumerate(train_loader):
            image_i, mask_i = data
            image = image_i.to(device)
            mask = mask_i.to(device)

            optimizer.zero_grad()

            output = model(image.float())

            train_loss = criteria(output.float, mask.long())

            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss
        train_losses.append(running_train_loss)

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                image_i, mask_i = data
                image = image_i.to(DEVICE)
                mask = mask_i.to(DEVICE)

                output = model(image.float())

                val_loss = criteria(output.float(), mask.long())
                running_val_loss += val_loss.item()

                iou = iou_score(output.float(), mask.long())
                running_iou += iou.item()

            val_losses.append(running_val_loss)
            iou.append(running_iou)
            print(f"Epoch : {e}, Train Loss : {running_train_loss}, Val Loss : {running_val_loss}, IOU : {running_iou}")


def main():
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
            ),
        ],
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
            ),
        ],
    )

    model = UNET().to(DEVICE)
    criteria = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    iou_score = JaccardIndex(task="multiclass", num_classes=2)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    train(train_loader, val_loader, DEVICE, optimizer, model, criteria, iou_score)

    # save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth"), model)


if __name__ == "__main__":
    main()
