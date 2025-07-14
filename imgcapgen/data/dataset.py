# imgcapgen/data/dataset.py
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from imgcapgen.utils.vocab import Vocabulary
import os

       
class CustomDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        captions_file=None, 
        dataframe=None, 
        transform=None, 
        freq_threshold=5, 
        vocab=None
    ):
        """
        Args:
            root_dir (str or Path): Directory with all the images.
            captions_file (str, optional): Path to CSV file with 'image' and 'caption' columns.
            dataframe (pd.DataFrame, optional): DataFrame with 'image' and 'caption' columns.
            transform (callable, optional): Optional transform to be applied on an image.
            freq_threshold (int, optional): Minimum frequency to keep words in vocab.
            vocab (Vocabulary, optional): Prebuilt Vocabulary object.
        """
        
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.ToTensor()

        if dataframe is not None:
            self.df = dataframe
        elif captions_file is not None:
            self.df = pd.read_csv(captions_file)
        else:
            raise ValueError("Either 'captions_file' or 'dataframe' must be provided.")

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        assert len(self.imgs) == len(self.captions), f"Mismatch: found {len(self.imgs)} images vs {len(self.captions)} captions"

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]

        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location).convert("RGB")
        img = self.transform(img)

        caption_vec = [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(caption_vec)
        