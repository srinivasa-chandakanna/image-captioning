# imgcapgen/scripts/train.py
import sys
import time
import json
import os
from pathlib import Path


## try:
##     project_root = Path(__file__).resolve().parents[2]  # from scripts → imgcapgen → repo root
## except NameError:
##     project_root = Path.cwd().resolve().parent
## 
## sys.path.append(str(project_root))
## 
## # ✅ Proper imports from new structure
## from imgcapgen.scripts.setup_paths import add_project_root_to_path, setup_paths
## add_project_root_to_path()

# External & project imports
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from collections import Counter
import spacy
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random
import csv
import nltk

nltk.download("punkt", force=True, quiet=True)
nltk.download("popular", force=True, quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.functional import cosine_similarity

from imgcapgen.utils.preprocess_captions import preprocess_captions
from imgcapgen.utils.common_utils import create_captions_dict, make_train_valid_dfs, get_transforms
from imgcapgen.utils.viz_utils import (
    sample_image_caption, show_image, show_image_batch_with_captions,
    plot_loss_curve, plot_bleu_curve, plot_all_metrics_curve, save_training_log_to_csv
)
from imgcapgen.utils.vocab import save_vocab
from imgcapgen.utils.collate import CapsCollate
from imgcapgen.data.dataset import CustomDataset

# Models
from imgcapgen.models import (
    ScratchCNN_LSTM,
    ResNet_LSTM,
    ResNetFineTune_LSTM,
    ResNetFineTune2_LSTM,
    ResNetFineTune2_Attention_LSTM
)


# ==========
# Validation function stays same
# ==========
def validate(model, data_loader, criterion, device, vocab_size, model_tag):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, captions in data_loader:
            images, captions = images.to(device), captions.to(device)
            outputs = model(images, captions)
            if outputs.size(1) == captions.size(1) - 1:
                loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].contiguous().view(-1))
            else:
                loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

            val_loss += loss.item()
    return val_loss / len(data_loader)

def save_checkpoint(model, optimizer, epoch, val_loss, model_tag, cfg, tag=""):
    """
    Save a checkpoint, but only keep best/final tagged files, not unique for every epoch.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss
    }

    if tag in ["best", "final"]:
        pointer_name = f"{model_tag}_{tag}_model.pth"
        pointer_path = Path(cfg.artifact_dir) / pointer_name
        torch.save(checkpoint, pointer_path)
        print(f"💾 Saved checkpoint: {pointer_path}")
        return pointer_path
    else:
        # optional: still save detailed if you ever want
        detailed_name = f"{model_tag}_epoch{epoch}_val{val_loss:.4f}.pth"
        detailed_path = Path(cfg.artifact_dir) / detailed_name
        torch.save(checkpoint, detailed_path)
        print(f"💾 Saved checkpoint: {detailed_path}")
        return detailed_path

        
# ==========
# Generic training loop
# ==========
def train_model(
    model, train_loader, valid_loader,
    criterion, optimizer,
    vocab_size, model_tag,
    device, train_dataset, valid_dataset, cfg,
    use_scheduler=False
):    
    mean = cfg.imagenet_norm.mean
    std = cfg.imagenet_norm.std
    num_epochs = cfg.epochs
    patience = cfg.patience
    best_val_loss = float('inf')
    best_epoch = -1
    time_epoch = []
    train_losses, val_losses = [], []
    bleu_scores, bleu2_scores, bleu3_scores, cosine_scores = [], [], [], []

    total_start_time = time.time()
    
    # ✅ Optional scheduler
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.factor, patience=2, verbose=True
        )
    else:
        scheduler = None

    # ✅ Setup TensorBoard writer
    #tb_log_dir = Path(cfg.artifact_dir) / "runs" / model_tag
    #writer = SummaryWriter(log_dir=tb_log_dir)
    #print(f"📈 TensorBoard logs will be saved to: {tb_log_dir}")
    
    # ✅ SBERT model to GPU if available
    sbert_device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens', device=sbert_device)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n📘 Epoch {epoch}/{num_epochs}")
        epoch_start_time = time.time()
        running_loss = 0.0
        # --------------------
        # TRAINING
        # --------------------
        model.train()
        for images, captions in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images, captions = images.to(device), captions.to(device)
            optimizer.zero_grad()                  # Zero the gradients.
            outputs = model(images, captions)      # Feed forward
            if outputs.size(1) == captions.size(1) - 1:
                loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].contiguous().view(-1))
            else:
                loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))  # Calculate the batch loss.
            loss.backward()                        # Backward pass.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()                       # Update the parameters in the optimizer.
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = validate(model, valid_loader, criterion, device, vocab_size, model_tag)
        if scheduler:
            scheduler.step(avg_val_loss)
            current_lrs = [pg['lr'] for pg in optimizer.param_groups]
            print(f"📉 LR scheduler adjusted: {[f'{lr:.6f}' for lr in current_lrs]}")
    
    
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # ✅ Log to TensorBoard
        #writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        #writer.add_scalar("Loss/Valid", avg_val_loss, epoch)
        #writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
        
        print(f"✅ Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"⏱️ Time: {(time.time() - epoch_start_time)/60:.2f} min")
        time_epoch.append((time.time() - epoch_start_time) / 60)

        # --------------------
        # EVALUATION ON FULL VAL SET
        # --------------------
        model.eval()
        all_references = []
        all_predictions = []
        all_true_sentences = []
        all_generated_sentences = []

        sample_indices = random.sample(range(len(valid_dataset)), min(3, len(valid_dataset)))
        printed_samples = 0

        print("🔍 Sample captions with BLEU & Cosine Similarity:")
        
        # Evaluate BLEU & Cosine on full validation set
        with torch.no_grad():
            for imgs, caps in tqdm(valid_loader, desc="Evaluating"):
                imgs = imgs.to(device)
                features = model.encoder(imgs)

                for i in range(features.size(0)):
                    feature = features[i].unsqueeze(0)
                    cap = caps[i]

                    gen_caption = model.decoder.generate_caption(feature, vocab=train_dataset.vocab)
                    generated_sentence = ' '.join(gen_caption)
                    true_sentence = train_dataset.vocab.decode(cap.tolist())

                    # corpus bleu style accumulation
                    all_references.append([word_tokenize(true_sentence)])
                    all_predictions.append(word_tokenize(generated_sentence))
                    all_true_sentences.append(true_sentence)
                    all_generated_sentences.append(generated_sentence)

                    global_idx = len(all_references) - 1
                    if global_idx in sample_indices:
                        printed_samples += 1
                        print(f"Sample {printed_samples}:")
                        print(f" True: {true_sentence}")
                        print(f" Pred: {generated_sentence}")
                        show_image(imgs[i].cpu(), mean, std, title="Sample Caption")

        # --------------------
        # COMPUTE CORPUS BLEU & COSINE
        # --------------------
        bleu = corpus_bleu(all_references, all_predictions)
        bleu2 = corpus_bleu(all_references, all_predictions, weights=(0.5, 0.5))
        bleu3 = corpus_bleu(all_references, all_predictions, weights=(0.33, 0.33, 0.34))

        true_vecs = sbert_model.encode(all_true_sentences, batch_size=32, convert_to_tensor=True)
        pred_vecs = sbert_model.encode(all_generated_sentences, batch_size=32, convert_to_tensor=True)
        cosine_scores_batch = cosine_similarity(true_vecs, pred_vecs).cpu().numpy()
        avg_cosine = cosine_scores_batch.mean()

        bleu_scores.append(bleu)
        bleu2_scores.append(bleu2)
        bleu3_scores.append(bleu3)
        cosine_scores.append(avg_cosine)

        print(f"🚀 Averages on FULL VAL SET -> BLEU: {bleu:.4f} | BLEU2: {bleu2:.4f} | BLEU3: {bleu3:.4f} | Cosine Similarity: {avg_cosine:.4f}")

        # --------------------
        # EARLY STOPPING
        # --------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_file = save_checkpoint(model, optimizer, epoch, best_val_loss, model_tag, cfg, tag="best")
            patience = cfg.patience
        else:
            patience -= 1
            print(f"⚠️ No improvement. Patience left: {patience}")
            if patience <= 0:
                print("⏹️ Early stopping triggered.")
                break

    # --------------------
    # SAVE FINAL MODEL & METADATA
    # --------------------
    final_model_file = save_checkpoint(model, optimizer, epoch, avg_val_loss, model_tag, cfg, tag="final")
    total_time = (time.time() - total_start_time) / 60
    print(f"\n📦 Final checkpoint saved at: {final_model_file}")
    print(f"🎯 Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f}")
    print(f"✅ Total Training Time: {total_time:.2f} min")

    # Metadata JSON
    metadata = {
        "model_tag": model_tag,
        "dataset": cfg.dataset,
        "epochs_completed": epoch,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "train_losses": train_losses,
        "valid_losses": val_losses,
        "bleu_scores": bleu_scores,
        "bleu2_scores": bleu2_scores,
        "bleu3_scores": bleu3_scores,
        "cosine_scores": cosine_scores,
        "best_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_model_file": str(best_model_file.name),
        "final_model_file": str(final_model_file.name),
        "time_per_epoch_minutes": time_epoch,
        "total_training_time_minutes": round(total_time, 2)
    }
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    metadata_native = convert_to_native(metadata)
    
    metadata_path = Path(cfg.output_dir) / f"{model_tag}_training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_native, f, indent=2)
    print(f"📄 Metadata saved: {metadata_path}")
    plot_loss_curve(train_losses, val_losses, cfg.output_dir, cfg.model_tag)
    plot_bleu_curve(bleu_scores, cfg.output_dir, cfg.model_tag)
    plot_all_metrics_curve(bleu_scores, bleu2_scores, bleu3_scores, cosine_scores, cfg.output_dir, cfg.model_tag)

    
    csv_path = Path(cfg.output_dir) / f"{model_tag}_training_log.csv"
    save_training_log_to_csv(csv_path,epochs=len(train_losses),train_losses=train_losses,
        val_losses=val_losses,time_per_epoch=time_epoch,bleu_scores=bleu_scores,bleu2_scores=bleu2_scores,
        bleu3_scores=bleu3_scores,cosine_scores=cosine_scores)

    # ✅ Close TensorBoard
    #writer.close()
    #print(f"📝 TensorBoard writer closed.")
    
    return model

# ==========
# Model-specific wrappers
# ==========
def train_scratchCNN_LSTM(cfg, train_dataset, train_loader, valid_dataset, valid_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(train_dataset.vocab)
    model = ScratchCNN_LSTM(cfg.embed_size, cfg.hidden_size, vocab_size, cfg.num_layers, cfg.dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
    return train_model(model, train_loader, valid_loader, criterion, optimizer, vocab_size, cfg.model_tag, device, train_dataset, valid_dataset, cfg)


def train_ResNetPreTrain_LSTM(cfg, train_dataset, train_loader, valid_dataset, valid_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(train_dataset.vocab)
    model = ResNet_LSTM(cfg.embed_size, cfg.hidden_size, vocab_size, cfg.num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
    return train_model(model, train_loader, valid_loader, criterion, optimizer, vocab_size, cfg.model_tag, device, train_dataset, valid_dataset, cfg)


def train_ResNetFineTune_LSTM(cfg, train_dataset, train_loader, valid_dataset, valid_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(train_dataset.vocab)
    model = ResNetFineTune_LSTM(cfg.embed_size, cfg.hidden_size, vocab_size, cfg.num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    # Different LR for encoder (CNN) vs decoder (LSTM)
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": float(cfg.encoder_lr)},
        {"params": model.decoder.parameters(), "lr": float(cfg.decoder_lr)}
    ])
    return train_model(model, train_loader, valid_loader, criterion, optimizer, vocab_size, cfg.model_tag, device, train_dataset, valid_dataset, cfg)

def train_ResNetFineTune2_LSTM(cfg, train_dataset, train_loader, valid_dataset, valid_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(train_dataset.vocab)
    model = ResNetFineTune2_LSTM(cfg.embed_size, cfg.hidden_size, vocab_size, cfg.num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    # Different LR for encoder (CNN) vs decoder (LSTM)
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": float(cfg.encoder_lr2)},
        {"params": model.decoder.parameters(), "lr": float(cfg.decoder_lr)}
    ])
    return train_model(model, train_loader, valid_loader, criterion, optimizer, vocab_size, cfg.model_tag, device, train_dataset, valid_dataset, cfg)

def train_ResNetFineTune2_Attention_LSTM(cfg, train_dataset, train_loader, valid_dataset, valid_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(train_dataset.vocab)

    if cfg.attention == 1:
        print("🚀 Using attention config: attention1")
        cfg_attention = cfg.attention1
    elif cfg.attention == 2:
        print("🚀 Using attention config: attention2")
        cfg_attention = cfg.attention2
    elif cfg.attention == 3:
        print("🚀 Using attention config: attention3")
        cfg_attention = cfg.attention3
    elif cfg.attention == 4:
        print("🚀 Using attention config: attention4")
        cfg_attention = cfg.attention4    
    else:
        print("🚀 Using default top-level config")
        cfg_attention = None

    if cfg_attention is not None:
        embed_size = cfg_attention.embed_size
        hidden_size = cfg_attention.hidden_size
        attn_dim = cfg_attention.attn_dim
        dropout = float(cfg_attention.dropout)
        encoder_lr = float(cfg_attention.encoder_lr)
        decoder_lr = float(cfg_attention.decoder_lr)
        weight_decay = float(cfg_attention.weight_decay)
        use_scheduler = getattr(cfg_attention, "use_scheduler", False)
    else:
        embed_size = cfg.embed_size
        hidden_size = cfg.hidden_size
        attn_dim = cfg.attn_dim
        dropout = float(cfg.dropout)
        encoder_lr = float(cfg.encoder_lr2)
        decoder_lr = float(cfg.decoder_lr)
        weight_decay = float(cfg.weight_decay)
        use_scheduler = False

    num_layers = cfg.num_layers

    # Build the model
    model = ResNetFineTune2_Attention_LSTM(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        attn_dim=attn_dim,
        num_layers=num_layers,
        drop_prob=dropout
    ).to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.decoder.parameters(), "lr": decoder_lr}
    ], weight_decay=weight_decay)

    return train_model(model, train_loader, valid_loader, criterion, optimizer, vocab_size, cfg.model_tag, device, train_dataset, valid_dataset, cfg, use_scheduler=use_scheduler)

# ==========
# Main driver
# ==========
def train(cfg, model_tag="Resnet_LSTM"):
    cfg.model_tag = model_tag

    # Preprocess captions & prepare dataset
    print("\n✅ Preprocess captions...")
    csv_out_file = Path(cfg.artifact_dir) / cfg.csv_filename
    df = preprocess_captions(Path(cfg.captions_path), csv_out_file)
    sample_image_caption(cfg.image_path, df)
    print("\n✅ Sample Captions Data:")
    print(df.head(5))
    print("\n✅ Sample Image & Captions:")
    sample_image_caption(cfg.image_path, df)
    print("\n✅ Preprocess captions: Completed")

    # Split data
    print("\n✅ Data Split to Train/Validation: Started")
    train_df, valid_df = make_train_valid_dfs(csv_out_file,cfg.debug)
    
    # Debug mode: reduce dataset size
    if cfg.debug:
        train_df = train_df.sample(n=min(2000, len(train_df)), random_state=42).reset_index(drop=True)
        valid_df = valid_df.sample(n=min(100, len(valid_df)), random_state=42).reset_index(drop=True)
        print(f"[DEBUG] Using {len(train_df)} train / {len(valid_df)} val samples")
    
    # Determine mean & std from cfg
    # Default to ImageNet normalization
    mean = cfg.imagenet_norm.mean
    std = cfg.imagenet_norm.std

    # Special case: scratchCNN_LSTM with dataset-specific norms
    if cfg.model_tag.lower() == "scratchcnn_lstm":
        if "flickr8k" in cfg.dataset.lower():
            mean = cfg.flickr8k_norm.mean
            std = cfg.flickr8k_norm.std
        elif "flickr30k" in cfg.dataset.lower():
            mean = cfg.flickr30k_norm.mean
            std = cfg.flickr30k_norm.std

    print(f"[Transforms] Using mean={mean}, std={std} for model={cfg.model_tag} on dataset={cfg.dataset}")

    #transforms = T.Compose([T.Resize((cfg.size, cfg.size)),T.ToTensor()])
        
    transforms = get_transforms(mean, std, cfg.size, mode="train")
                
    train_dataset = CustomDataset(cfg.image_path, dataframe=train_df, transform=transforms, vocab=None)
    valid_dataset = CustomDataset(cfg.image_path, dataframe=valid_df, transform=transforms, vocab=train_dataset.vocab)
    save_vocab(train_dataset.vocab, cfg.artifact_dir)
    
    num_unique_tokens = len(train_dataset.vocab.stoi)
    print(f"Number of unique tokens in vocab: {num_unique_tokens}")
    
    special_tokens = {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}
    num_actual_words = len([word for word in train_dataset.vocab.stoi if word not in special_tokens])
    print(f"Number of real words in vocab: {num_actual_words}")
    
    print("\n✅ Sample Image & Tokens:")
    idx = random.randint(0, len(train_dataset) - 1)
    img, caps = train_dataset[idx]
    show_image(img,mean,std,"Sample Image")
    print("Token :",caps)
    print("Sentence: ")
    print([train_dataset.vocab.itos[token] for token in caps.tolist()])
    
    
    # Data loaders
    pad_idx = train_dataset.vocab.stoi["<PAD>"]
    collate_fn = CapsCollate(pad_idx=pad_idx, batch_first=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False, collate_fn=collate_fn)
     
    # Get a batch from DataLoader 
    print("\n✅ Sample Images in Batch:")
    dataiter = iter(train_loader)
    images, captions = next(dataiter)
    show_image_batch_with_captions(images, captions, train_dataset.vocab, mean, std, num_images=cfg.num_sample_retrieval)

    def get_trainer(model_key):
        if model_key.startswith("resnetfinetune2_attention") and model_key.endswith("_lstm"):
            return train_ResNetFineTune2_Attention_LSTM
        return trainers.get(model_key, None)
        
    # Trainer dispatch
    trainers = {
    "scratchcnn_lstm": train_scratchCNN_LSTM,
    "resnet_lstm": train_ResNetPreTrain_LSTM,
    "resnetfinetune_lstm": train_ResNetFineTune_LSTM,
    "resnetfinetune2_lstm": train_ResNetFineTune2_LSTM,
    #"resnetfinetune2_attention_lstm": train_ResNetFineTune2_Attention_LSTM,
    #"resnetfinetune2_attention2_lstm": train_ResNetFineTune2_Attention_LSTM,
    #"resnetfinetune2_attention3_lstm": train_ResNetFineTune2_Attention_LSTM,
    #"resnetfinetune2_attention4_lstm": train_ResNetFineTune2_Attention_LSTM,
    }
    
    model_key = model_tag.lower()
    #trainer_func = trainers.get(model_key)
    trainer_func = get_trainer(model_key)
    
    if trainer_func is None:
        raise ValueError(f"Trainer not found for model: {model_key}")

    
    if trainer_func:
        print(f"\n🚀 Starting training for: {model_tag}")
        trained_model = trainer_func(cfg, train_dataset, train_loader, valid_dataset, valid_loader)
        print(f"\n✅ {model_tag} training complete.")
        return trained_model
    else:
        print(f"⚠️ Unknown model '{model_tag}'. Available: {list(trainers.keys())}")
        return None

def run_training_pipeline(config_path, model_tag="Resnet_LSTM"):
    from imgcapgen import config as config_module
    from imgcapgen.scripts import setup_paths

    # Load YAML config into DotConfig
    cfg = config_module.load_config(config_path)
    
    # Setup paths dynamically
    setup_paths.setup_paths(cfg)

    # Dispatch to training
    return train(cfg, model_tag)
