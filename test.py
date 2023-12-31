import argparse

import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from utils import load_checkpoint
from unet import UNET


parser = argparse.ArgumentParser(description="Test UNET model on an image and mask.")
parser.add_argument("--image_path", 
                    default='./data/test/image/1.png',
                    type=str, 
                    help="path to the image")
parser.add_argument("--mask_path", 
                    default='./data/test/mask/1.png',
                    type=str, 
                    help="path to the mask")
args = parser.parse_args()


def testing(model, image_path, mask_path):
    # Load the image and mask
    image_path = image_path  
    image = Image.open(image_path)

    mask_path = mask_path
    mask = Image.open(mask_path)

    # Preprocess the image 
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    # Make predictions
    model.eval()
    with torch.no_grad():
        output_mask = model(image_tensor)
        output_mask = output_mask.squeeze(0)

    # Postprocess the output
    output_mask = torch.sigmoid(output_mask)
    threshold = 0.5 # threshold for pixel values
    binary_output_mask = (output_mask > threshold).float()

    # Plot the original image, model output, and label
    plt.figure(figsize=(12, 4))

    # Plot the original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")

    # Display the actual label
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Original Mask")

    # Plot the model output 
    plt.subplot(1, 3, 3)
    plt.imshow(binary_output_mask[0], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.title("Model Output")

    plt.tight_layout()
    
    # Save the combined image as "prediction.jpg"
    plt.savefig('prediction.jpg')
    plt.show()


def main():
    # Load the model
    model = UNET()
    checkpoint = torch.load('./my_checkpoint.pth') # path to checkpoint
    load_checkpoint(checkpoint, model)

    # Test the model
    testing(model, image_path=args.image_path, mask_path=args.mask_path)


if __name__ == "__main__":
    main()












