import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from unet import UNET

from loss import loss_function, dice_metric
from utils import load_checkpoint, save_checkpoint, get_loaders

from config import (LEARNING_RATE, DEVICE, BATCH_SIZE, NUM_WORKERS, NUM_EPOCHS, IMAGE_HEIGHT, IMAGE_WIDTH, PIN_MEMORY,
                    LOAD_MODEL, TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR)


def train(train_loader, val_loader, device, optimizer, model, criteria):
    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []
    for e in tqdm(range(NUM_EPOCHS)):
        model.train()
        train_step = 0

        running_train_loss, running_val_loss = 0, 0
        running_train_metric, running_val_metric = 0, 0

        for i, data in enumerate(train_loader):
            train_step += 1

            image_i, mask_i = data
            image = image_i.to(device)
            mask = mask_i.to(device)

            optimizer.zero_grad()

            output = model(image.float())

            train_loss = criteria(output.float(), mask.float())

            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()

            train_metric = dice_metric(output.float(), mask.float())
            running_train_metric += train_metric

        train_losses.append(running_train_loss / train_step)
        train_metrics.append(running_train_metric / train_step)

        model.eval()
        test_step = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                test_step += 1

                image_i, mask_i = data
                image = image_i.to(DEVICE)
                mask = mask_i.to(DEVICE)

                output = model(image.float())

                val_loss = criteria(output.float(), mask.float())
                running_val_loss += val_loss.item()

                val_metric = dice_metric(output.float(), mask.float())
                running_val_metric += val_metric

            val_losses.append(running_val_loss / test_step)
            val_metrics.append(running_val_metric / test_step)

            print(f"\n\nEpoch : {e}")
            print("-" * 20)
            print(f"Train Loss : {running_train_loss / train_step}, Val Loss : {running_val_loss / test_step}")
            print(f"Train Metric : {running_train_metric / train_step}, Val Metric : {running_val_metric / test_step}")


def main():
    train_transform = transforms.Compose(
        [

            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.Normalize(
                mean=[0.0],
                std=[1.0],
            ),
        ],
    )

    val_transforms = transforms.Compose(
        [

            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.Normalize(
                mean=[0.0],
                std=[1.0],
            ),
        ],
    )

    model = UNET().to(DEVICE)
    criteria = loss_function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    train(train_loader, val_loader, DEVICE, optimizer, model, criteria)

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
