# imgcapgen/utils/vocab.py

import pickle
import pandas as pd
from collections import Counter
from pathlib import Path
import spacy

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.spacy_eng = spacy.load("en_core_web_sm")

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        return [token.text.lower() for token in self.spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

    def decode(self, token_list):
        return ' '.join([
            self.itos.get(idx, "<UNK>")
            for idx in token_list
            if self.itos.get(idx, "<UNK>") not in ["<PAD>", "<SOS>", "<EOS>"]
        ])
                         
def save_vocab(vocab, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save pickle
    with open(save_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # Save CSV
    vocab_entries = [(word, idx) for word, idx in vocab.stoi.items()]
    vocab_df = pd.DataFrame(vocab_entries, columns=["word", "index"])
    vocab_df["itos_check"] = vocab_df["index"].map(vocab.itos)
    vocab_df.to_csv(save_dir / "vocab.csv", index=False)
    print(f"✅ Saved vocab to: {save_dir}")


def load_vocab(save_dir):
    vocab_path = Path(save_dir) / "vocab.pkl"
    with open(vocab_path, "rb") as f:
        loaded_vocab = pickle.load(f)
    print(f"✅ Loaded vocab from: {vocab_path}")
    return loaded_vocab