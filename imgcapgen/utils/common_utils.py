# imgcapgen/utils/common_utils.py

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
import albumentations as A
import torch.nn as nn
import torchvision.transforms as T
from pathlib import Path

#####################################################################################
# === Used for captioning ===

def create_captions_dict(captions_csv_file):
    df = pd.read_csv(captions_csv_file)
    captions_dict = defaultdict(list)
    for _, row in df.iterrows():
        captions_dict[row['image']].append(row['caption'])
    return captions_dict   
    
    
# === Data Prep Functions ===
def make_train_valid_dfs(csv_path, debug=False):
    """
    Split CSV into train and valid dataframes.

    Constructs the CSV path from output_dir / csv_filename.
    """
    csv_path = Path(csv_path)

    dataframe = pd.read_csv(csv_path)
    max_id = dataframe["id"].max() + 1 if not debug else 100
    image_ids = np.arange(0, max_id)

    np.random.seed(42)
    valid_ids = np.random.choice(image_ids, size=int(0.2 * len(image_ids)), replace=False)
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]

    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)

    # Drop the 'caption_number' and 'id' columns if they exist
    train_dataframe = train_dataframe.drop(columns=['caption_number', 'id'], errors='ignore')
    valid_dataframe = valid_dataframe.drop(columns=['caption_number', 'id'], errors='ignore')

    return train_dataframe, valid_dataframe  
    
def get_transforms(mean, std, size, mode="train"):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])



def prepare_unseen_data(captions_csv_path, image_dir_path):
    """
    Prepare unseen data in the format:
    [
        ("/path/to/image1.jpg", ["caption1", "caption2", ...]),
        ...
    ]
    """
    captions_csv_path = Path(captions_csv_path)
    image_dir_path = Path(image_dir_path)

    # Load CSV
    captions_df = pd.read_csv(captions_csv_path)

    # Group captions by image
    captions_dict = defaultdict(list)
    for _, row in captions_df.iterrows():
        image_name = row["image"]
        caption_text = row["caption"]
        captions_dict[image_name].append(caption_text)

    # Build unseen_data
    unseen_data = []
    for image_name, captions in captions_dict.items():
        full_image_path = str(image_dir_path / image_name)
        unseen_data.append( (full_image_path, captions) )

    print(f"✅ Prepared unseen_data with {len(unseen_data)} images")
    return unseen_data    
##################################################################################### 

#####################################################################################
# === Simple utility classes & functions ===

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

#####################################################################################
# === Used for CLIP training ===

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def build_loaders(dataframe, tokenizer, size, mean, std, batch_size, num_workers, device, mode):
    from imgcapgen.data.clip_dataset import CLIPDataset  # adjust if dataset changes
    transforms = get_transforms(mean, std, size=size, mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

# === Training & Validation Functions ===

def train_epoch(model, train_loader, optimizer, lr_scheduler, step, device):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter


def valid_epoch(model, valid_loader, device):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    return loss_meter