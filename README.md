# Image Captioning 🖼️📝
A modular deep learning framework to train **Image Captioning** models using CNN (scratch / ResNet) + LSTM architectures on the **Flickr8k** and **Flickr30k** datasets.  
Supports both **Google Drive** and **Kaggle** data workflows, and logs training, BLEU, cosine similarity metrics.

---

## 🚀 Key Features
- Custom **CNN + LSTM**, **ResNet + LSTM**, and **fine-tuned ResNet + LSTM** models
- BLEU-1, BLEU-2, BLEU-3, SBERT cosine similarity evaluation
- Early stopping & robust checkpointing
- Config-driven (YAML) pipeline for easy dataset/model switching
- Visualization of sample captions, metrics & loss curves
- Works seamlessly in **Google Colab** with Drive or locally with KaggleHub.

---

## 📂 Directory Structure
```
image-captioning/
│
├── imgcapgen/
│   ├── config/           # YAML configs & loader
│   ├── data/             # Dataset classes
│   ├── models/           # CNN-LSTM models
│   ├── scripts/          # setup_paths.py, train.py
│   └── utils/            # preprocessing, vocab, viz etc.
│
├── outputs/              # Evaluation metrics, plots, logs
├── artifacts/            # Saved model checkpoints
├── Run/                  # Notebooks for training & experiments
│
├── requirements.txt
├── setup.py
├── LICENSE.txt
└── README.md
```

---

## ⚙️ Installation

### 🐍 Local / Kaggle
```bash
git clone https://github.com/yourusername/image-captioning.git
cd image-captioning
pip install -r requirements.txt
```

### ☁️ Google Colab (recommended for Drive workflow)
- Clone or upload your repo inside Google Drive.
- Open `Run/run_train.ipynb`, adjust `root_folder_name`, and run cells.

---

## 📑 Config YAML
Supports multiple datasets with separate YAML files.  
Just adjust `dataset`, `data_source`, `dataset_url`, `image_path`, `captions_path`.

Example for **Flickr8k (Kaggle)**:
```yaml
dataset: "flickr8k"
data_source: "kaggle"
dataset_url: "adityajn105"
image_path: "Images"
captions_path: "captions.txt"
artifact_dir: "artifacts"
output_dir: "outputs"
...
```
For **Google Drive** change to:
```yaml
data_source: "google_drive"
dataset_url: "flickr8k_dataset"
image_path: "Images"
captions_path: "captions.txt"
```

---

## 🚀 Usage

### ✅ Colab Notebook
Inside `Run/` you’ll find `run_train.ipynb`:

```python
from imgcapgen.config.config import cfg
from imgcapgen.scripts.setup_paths import setup_paths
from scripts.train import train

selected_dataset = "flickr8k"
selected_model = "ScratchCNN_LSTM"

cfg.dataset = selected_dataset
setup_paths(cfg)

trained_model = train(selected_model)
```

---

### ✅ Direct Python
```bash
python imgcapgen/scripts/train.py
```
(Default model is `Resnet_LSTM`, change in `train.py` or pass in notebook).

---

## 📝 Outputs
- 🗂️ `artifacts/{dataset}/`: model checkpoints (`*_epoch*_val*.pth`), best & final models.
- 📈 `outputs/{dataset}/`: 
  - JSON metadata logs
  - CSV logs with losses & BLEU
  - Loss & BLEU plots.

---

## 📜 License
This project is licensed under the terms of the `MIT License`.  
See [`LICENSE.txt`](LICENSE.txt) for details.

---

## 🙌 Acknowledgements
- Inspired by standard CNN-LSTM image captioning pipelines.
- Uses **SentenceTransformers** for cosine similarity.
- Thanks to **Flickr8k** and **Flickr30k** for datasets.

---

🚀 **Happy Captioning!**
