# imgcapgen/utils/viz_utils.py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from typing import List
import torch
import textwrap
import numpy as np
import csv
import random
from PIL import Image


def sample_image_caption_old(image_path, df):
    """Randomly pick an image and print its next 5 captions."""
    data_idx = random.randint(0, max(0, len(df) - 5))
    img_path = Path(image_path) / df.iloc[data_idx, 0]
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    for i in range(data_idx, data_idx + 5):
        print(f"Caption - {df.iloc[i, 2]}")

def sample_image_caption(image_path, df):
    """Randomly pick a valid image and show its captions."""
    for _ in range(5):  # try up to 5 attempts
        data_idx = random.randint(0, max(0, len(df) - 5))
        img_path = Path(image_path) / df.iloc[data_idx, 0]

        try:
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.axis("off")
            plt.title("Sample Image with Captions")
            plt.show()

            for i in range(data_idx, data_idx + 5):
                print(f"Caption - {df.iloc[i, 2]}")
            return

        except Exception as e:
            print(f"⚠️ Skipping corrupted image: {img_path}")
            print(f"🛠️ Error: {e}")

    print("❌ Could not display a valid image after multiple attempts.")




def denormalize_image(img_tensor, mean, std):
    """
    img_tensor: Tensor [C, H, W] normalized
    mean, std: lists of 3 floats
    Returns: tensor [C, H, W] denormalized
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return img_tensor * std + mean
    
    


def show_image(inp, mean, std, title=None):
    """Imshow for Tensor with denormalization"""
    inp = denormalize_image(inp, mean, std)
    inp = inp.numpy().transpose((1,2,0))  # CxHxW → HxWxC
    inp = np.clip(inp, 0, 1)  # ensure valid range for plt.imshow
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.pause(0.001)

def show_image_batch_with_captions(images: torch.Tensor, captions: torch.Tensor, vocab, mean, std, num_images=5):
    num_images = min(num_images, images.size(0))
    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
    fig.tight_layout(pad=4.0)

    for i in range(num_images):
        img = images[i]
        cap = captions[i]

        # Decode caption
        caption_tokens = [vocab.itos[token] for token in cap.tolist()]
        eos_index = caption_tokens.index('<EOS>') if '<EOS>' in caption_tokens else len(caption_tokens)
        caption_text = ' '.join(caption_tokens[1:eos_index])
        wrapped_caption = textwrap.fill(caption_text, width=30)

        # Denormalize image
        img = denormalize_image(img, mean, std)
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        # Plot
        ax = axes[i] if num_images > 1 else axes
        ax.imshow(img_np)
        ax.set_title(wrapped_caption, fontsize=8)
        ax.axis("off")

    plt.show()
    plt.close()

  

def plot_loss_curve(train_losses, valid_losses, output_dir, model_tag):
    
    # Build output path using model tag and dataset
    plot_path = Path(output_dir) / f"{model_tag}_loss_curve.png"

    # Prepare epoch range starting from 1
    epochs = np.arange(1, len(train_losses) + 1)

    # Create plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, valid_losses, label="Valid Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.xticks(epochs)  # force integer ticks
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\n✅ Loss curve saved to: {plot_path}") 
    plt.show()
    plt.close()       


def plot_bleu_curve(bleu_scores, output_dir, model_tag):
    bleu_plot_path = Path(output_dir) / f"{model_tag}_bleu_curve.png"
    epochs = np.arange(1, len(bleu_scores) + 1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs, bleu_scores, label="BLEU", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("BLEU Score")
    plt.title("BLEU Score over Epochs")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(bleu_plot_path)
    print(f"✅ BLEU curve saved to: {bleu_plot_path}")
    plt.show()
    plt.close()
    
def plot_all_metrics_curve(bleu, bleu2, bleu3, cosine, output_dir, model_tag):
    all_plot_path = Path(output_dir) / f"{model_tag}_all_metrics_curve.png"
    epochs = np.arange(1, len(bleu) + 1)
    plt.figure(figsize=(10,6))
    plt.plot(epochs, bleu, label="BLEU")
    plt.plot(epochs, bleu2, label="BLEU2")
    plt.plot(epochs, bleu3, label="BLEU3")
    plt.plot(epochs, cosine, label="Cosine Similarity")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("BLEU & Cosine Similarity over Epochs")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(all_plot_path)
    print(f"✅ BLEU curve saved to: {all_plot_path}")
    plt.show()
    plt.close()


def save_training_log_to_csv(csv_path,epochs,train_losses,val_losses,time_per_epoch,bleu_scores,bleu2_scores,bleu3_scores,cosine_scores):
    header = [
        "epoch", "train_loss", "val_loss", "time_minutes",
        "bleu", "bleu2", "bleu3", "cosine_similarity"
    ]
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(epochs):
            writer.writerow([
                i + 1,
                train_losses[i],
                val_losses[i],
                time_per_epoch[i],
                bleu_scores[i],
                bleu2_scores[i],
                bleu3_scores[i],
                cosine_scores[i],
            ])
    print(f"📊 Training log CSV saved to: {csv_path}")


    
