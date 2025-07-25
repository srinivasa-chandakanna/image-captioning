import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.table as tbl
import seaborn as sns
import textwrap
import torch
import torchvision.transforms as T
from PIL import Image
import pandas as pd
from tqdm import tqdm
import nltk

nltk.download("punkt", force=True, quiet=True)
nltk.download("popular", force=True, quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer

from imgcapgen.utils.vocab import load_vocab

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_class, embed_size, hidden_size, vocab_size, model_path, num_layers, drop_prob, attn_dim, Attention):
    if Attention:
        model = model_class(embed_size, hidden_size, vocab_size, attn_dim, num_layers, drop_prob)
    else:
        model = model_class(embed_size, hidden_size, vocab_size, num_layers, drop_prob)
    
    print(f"model : {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def predict_caption(model, image_path, vocab, mean, std):
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model.encoder(image)
        tokens = model.decoder.generate_caption(features, vocab=vocab)
    return " ".join(tokens)


def evaluate_single_model_on_data(model, test_data, vocab, sbert_model, mean, std):
    bleu4_scores, bleu2_scores, bleu3_scores, meteor_scores, cosine_scores = [], [], [], [], []
    all_true_sentences, all_pred_sentences = [], []
    predictions = []

    for img_path, gt_captions in tqdm(test_data, desc="Evaluating model"):
        pred_caption = predict_caption(model, img_path, vocab, mean, std)

        references_tokens = [ref.split() for ref in gt_captions]
        hypothesis = pred_caption.split()

        bleu4 = corpus_bleu([references_tokens], [hypothesis])
        bleu2 = corpus_bleu([references_tokens], [hypothesis], weights=(0.5,0.5))
        bleu3 = corpus_bleu([references_tokens], [hypothesis], weights=(0.33,0.33,0.34))
        meteor = meteor_score(references_tokens, hypothesis)

        bleu4_scores.append(bleu4)
        bleu2_scores.append(bleu2)
        bleu3_scores.append(bleu3)
        meteor_scores.append(meteor)

        all_true_sentences.append(gt_captions[0])
        all_pred_sentences.append(pred_caption)

        # Temporarily add cosine=0, will update after SBERT
        predictions.append((img_path, pred_caption, bleu4, 0.0))

    # Now compute cosine similarities in batch
    true_vecs = sbert_model.encode(all_true_sentences, batch_size=32, convert_to_tensor=True)
    pred_vecs = sbert_model.encode(all_pred_sentences, batch_size=32, convert_to_tensor=True)
    cosines = cosine_similarity(true_vecs, pred_vecs).cpu().numpy()

    # Update predictions with actual cosine scores
    updated_predictions = []
    for (img_path, pred_caption, bleu4, _), cosine in zip(predictions, cosines):
        updated_predictions.append((img_path, pred_caption, bleu4, cosine))

    return updated_predictions, bleu4_scores, bleu2_scores, bleu3_scores, meteor_scores, cosines




def compute_avg_metrics(bleu4, bleu2, bleu3, meteor, cosine):
    return (
        round(sum(bleu4)/len(bleu4),4),
        round(sum(bleu2)/len(bleu2),4),
        round(sum(bleu3)/len(bleu3),4),
        round(sum(meteor)/len(meteor),4),
        round(sum(cosine)/len(cosine),4)
    )


def save_metrics_csv(metrics_list, output_dir):
    df = pd.DataFrame(metrics_list)
    csv_path = os.path.join(output_dir, "model_unseen_comparison_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV: {csv_path}")
    return csv_path


def create_html_report(selected_samples, sample_outputs, output_dir):
    html = "<h1>Unseen Data Caption Comparison</h1>"
    for img_path, gt in selected_samples:
        html += f"<h3>{os.path.basename(img_path)}</h3><img src='{img_path}' width='300'><br><b>GT:</b> {gt}<br>"
        for name, preds in sample_outputs:
            for p_img, _, pred_caption in preds:
                if p_img==img_path:
                    html += f"<b>{name}:</b> {pred_caption}<br>"
        html += "<hr>"
    path = os.path.join(output_dir, "sample_comparison.html")
    with open(path, "w") as f: f.write(html)
    print(f"✅ HTML saved: {path}")
    return path



def wrap_text_auto(text, base_width, scale_factor=1.0):
    """Wraps text to approximately fit computed width."""
    width = max(20, int(base_width * scale_factor))
    return "\n".join(textwrap.wrap(text, width=width))


def plot_caption_samples_with_table(selected_samples, sample_outputs, output_dir=None,
                                    fig_width=16, fig_height_per_sample=5,
                                    font_size=12, scale_by_caption=True):
    """
    Plots image + table with GT captions, predicted captions, and BLEU/Cosine scores.
    Automatically increases row height for prediction rows.
    """
    num_samples = len(selected_samples)
    fig_height = num_samples * fig_height_per_sample
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(num_samples, 2, width_ratios=[1, 2])

    save_path = os.path.join(output_dir, "sample_captions_table.png") if output_dir else None

    base_wrap_width = int((fig_width * 10) / (font_size / 10))
    base_wrap_width = max(20, base_wrap_width)

    if scale_by_caption:
        all_caps = []
        for _, gt_captions in selected_samples:
            all_caps.extend(gt_captions)
        avg_cap_len = np.mean([len(c) for c in all_caps]) if all_caps else 50
        scale_factor = min(2.0, avg_cap_len / 50.0)
    else:
        scale_factor = 1.0

    print(f"Auto wrap width: base={base_wrap_width}, scale={scale_factor:.2f} => final ≈{int(base_wrap_width*scale_factor)}")

    for i, (img_path, gt_captions) in enumerate(selected_samples):
        ax_img = plt.subplot(gs[i, 0])
        img = plt.imread(img_path)
        ax_img.imshow(img)
        ax_img.axis('off')

        table_data = []
        prediction_row_indices = []  # to track which rows are predictions

        for idx, gt in enumerate(gt_captions[:5]):
            table_data.append([f"GT {idx+1}", wrap_text_auto(gt, base_wrap_width, scale_factor)])


        for name, preds in sample_outputs:
            for p_img, pred_caption, bleu4, bleu2, bleu3, meteor, cosine in preds:
                if p_img == img_path:
                    caption_text = wrap_text_auto(pred_caption, base_wrap_width, scale_factor)
                    caption_with_scores = (f"{caption_text}\n"
                                           f"BLEU2: {bleu2:.2f} | BLEU3: {bleu3:.2f} | BLEU4: {bleu4:.2f} | "
                                           f"METEOR: {meteor:.2f} | Cos: {cosine:.2f}")
                    table_data.append([name, caption_with_scores])
                    prediction_row_indices.append(len(table_data) - 1)  # mark this row index

        ax_table = plt.subplot(gs[i, 1])
        ax_table.axis('off')
        table = ax_table.table(cellText=table_data,
                               colLabels=["Type", "Caption"],
                               loc='center',
                               cellLoc='left',
                               colWidths=[0.25, 0.75])

        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1.2, 1.5)

        # Make prediction rows taller
        for row_idx in prediction_row_indices:
            for col_idx in range(2):  # 2 columns
                if (row_idx+1, col_idx) in table._cells:  # +1 because header is at (0, _)
                    cell = table._cells[(row_idx+1, col_idx)]
                    cell.set_height(0.10)  # adjust height as needed

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()




def plot_metric_histograms(b, b2, b3, m, c, output_dir, model_name="Model"):
    plt.figure(figsize=(16,10))

    plt.subplot(2,3,1)
    sns.histplot(b2, kde=True, bins=15, color='orange')
    plt.title(f"{model_name} BLEU-2 Distribution")
    plt.xlabel("BLEU-2 Score")

    plt.subplot(2,3,2)
    sns.histplot(b3, kde=True, bins=15, color='green')
    plt.title(f"{model_name} BLEU-3 Distribution")
    plt.xlabel("BLEU-3 Score")

    plt.subplot(2,3,3)
    sns.histplot(b, kde=True, bins=15, color='skyblue')
    plt.title(f"{model_name} BLEU-4 Distribution")
    plt.xlabel("BLEU-4 Score")

    plt.subplot(2,3,4)
    sns.histplot(m, kde=True, bins=15, color='purple')
    plt.title(f"{model_name} METEOR Distribution")
    plt.xlabel("METEOR Score")

    plt.subplot(2,3,5)
    sns.histplot(c, kde=True, bins=15, color='red')
    plt.title(f"{model_name} Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")

    plt.tight_layout()
    path = os.path.join(output_dir, f"{model_name}_metric_histograms.png")
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()
    print(f"✅ Histogram plot saved: {path}")
    return path


def plot_all_metrics_grid(
    all_metrics_dict,  # dict: {"BLEU4": all_b1, "BLEU2": all_b2, ...}
    model_names,
    output_dir
):
    """
    all_metrics_dict: dict of metric_name -> list of lists, each inner list is values for one model
    model_names: list of model names
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    metric_names = list(all_metrics_dict.keys())
    num_metrics = len(metric_names)
    num_models = len(model_names)

    fig, axs = plt.subplots(num_metrics, num_models, figsize=(5*num_models, 4*num_metrics), sharey='row')

    # If only one row or one column, axs needs to be adjusted to be a 2D array
    if num_metrics == 1:
        axs = np.expand_dims(axs, axis=0)
    if num_models == 1:
        axs = np.expand_dims(axs, axis=1)

    colors = ['tab:orange', 'tab:green', 'tab:blue', 'tab:red', 'tab:purple']

    for row_idx, metric_name in enumerate(metric_names):
        all_metrics = all_metrics_dict[metric_name]
        for col_idx, (metric_values, model_name) in enumerate(zip(all_metrics, model_names)):
            ax = axs[row_idx, col_idx]
            # Histogram
            ax.hist(metric_values, bins=20, color=colors[col_idx % len(colors)],
                    edgecolor='black', alpha=0.6, density=True)
            # KDE overlay
            sns.kdeplot(metric_values, ax=ax, color='black', linewidth=1.5)

            if row_idx == 0:
                ax.set_title(f"{model_name}", fontsize=14)
            if col_idx == 0:
                ax.set_ylabel(f"{metric_name}\nDensity", fontsize=14)
            ax.set_xlabel(metric_name)

    #plt.suptitle("Histograms + KDEs of Metrics Across Models", fontsize=18, y=0.92)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, "all_metrics_grid_histograms_kde.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()
    print(f"✅ Saved combined plot to {save_path}")





def plot_comparison_metrics(metrics_list, output_dir):
    labels = [m["Model"] for m in metrics_list]
    bleu4  = [m["Avg BLEU4"] for m in metrics_list]
    bleu2 = [m["Avg BLEU2"] for m in metrics_list]
    bleu3 = [m["Avg BLEU3"] for m in metrics_list]
    meteor= [m["Avg METEOR"] for m in metrics_list]
    cosine= [m["Avg Cosine"] for m in metrics_list]

    y = np.arange(len(labels))
    height = 0.15

    fig, ax = plt.subplots(figsize=(10,8))

    # Add shaded background bands and horizontal lines
    for i in range(len(labels)):
        if i % 2 == 0:
            ax.axhspan(i - 0.4, i + 0.4, facecolor='lightgrey', alpha=0.3)
        ax.axhline(i + 0.4, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axhline(i - 0.4, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)

    # Draw horizontal bars

    bars1 = ax.barh(y - 2*height, bleu2, height, label='BLEU2', color='orange')
    bars2 = ax.barh(y - height, bleu3, height, label='BLEU3', color='green')
    bars3 = ax.barh(y , bleu4, height, label='BLEU4', color='skyblue')
    bars4 = ax.barh(y + height, meteor, height, label='METEOR', color='purple')
    bars5 = ax.barh(y + 2*height, cosine, height, label='Cosine', color='red')

    ax.set_xlabel('Scores')
    ax.set_title('Metrics Across Models')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.legend()

    # Annotate bars with color-coded text
    def add_values(bars, color):
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.2f}',
                        xy=(width, bar.get_y() + bar.get_height()/2),
                        xytext=(3,0),
                        textcoords="offset points",
                        ha='left', va='center',
                        fontsize=8, color=color)


    add_values(bars1, 'orange')
    add_values(bars2, 'green')
    add_values(bars3, 'deepskyblue')
    add_values(bars4, 'purple')
    add_values(bars5, 'red')

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_metrics_bars_horizontal.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"✅ Horizontal bar plot with shaded bands & lines saved: {path}")
    return path



def plot_comparison_line(metrics_list, output_dir):
    metrics_names = ["BLEU2","BLEU3","BLEU4","METEOR","Cosine"]
    plt.figure(figsize=(10,6))
    for m in metrics_list:
        scores = [m["Avg BLEU2"], m["Avg BLEU3"], m["Avg BLEU4"], m["Avg METEOR"], m["Avg Cosine"]]
        plt.plot(metrics_names, scores, marker='o', label=m["Model"])
    plt.ylabel("Scores"); plt.title("Line Plot of Metrics"); plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_metrics_line.png")
    plt.savefig(path); plt.show()
    print(f"✅ Line plot saved: {path}")
    return path

def evaluate_models_on_unseen(yaml_path, available_models, test_data,trained_dataset="flickr8k", num_samples_to_plot=5, root_dir="."):
    cfg = load_yaml_config(yaml_path)

    vocab_path = os.path.join(root_dir, "artifacts", trained_dataset)
    output_dir = os.path.join(root_dir, "validation_output", f"{trained_dataset}_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    vocab = load_vocab(vocab_path)
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens', device=DEVICE)

    metrics_list, sample_outputs = [], []
    all_b, all_b2, all_b3, all_meteor, all_cosine = [], [], [], [], []
    model_names = []

    for name, (tag, model_class) in available_models.items():
        print(f"\n🔍 Evaluating {name}")
        
        Attention = False
        # Use specialized config for attention models
        if "attention2" in tag:
            sub_cfg = cfg["attention2"]
            Attention = True
        elif "attention3" in tag:
            sub_cfg = cfg["attention3"]
            Attention = True
        elif "attention4" in tag:
            sub_cfg = cfg["attention4"]
            Attention = True
        elif "attention1" in tag:
            sub_cfg = cfg["attention1"]
            Attention = True
        else:
            sub_cfg = cfg  # defaults for normal models
            Attention = False
     
        embed, hidden = sub_cfg["embed_size"], sub_cfg["hidden_size"]
        drop_prob = sub_cfg["dropout"]
        attn_dim = sub_cfg["attn_dim"]
        num_layers = cfg["num_layers"]
        mean, std = cfg["imagenet_norm"]["mean"], cfg["imagenet_norm"]["std"]
     
        print(f"📌 Loading with embed={embed}, hidden={hidden}, dropout={drop_prob}, attn_dim={attn_dim}")        
        
        
        model_path = os.path.join(root_dir, "artifacts", trained_dataset, f"{tag}_best_model.pth")

        model = load_model(model_class, embed, hidden, len(vocab), model_path, num_layers, drop_prob, attn_dim, Attention)

        preds, b, b2, b3, m, c = evaluate_single_model_on_data(model, test_data, vocab, sbert_model, mean, std)
        avg_b, avg_b2, avg_b3, avg_m, avg_c = compute_avg_metrics(b, b2, b3, m, c)

        metrics_list.append({
            "Model": name,  "Avg BLEU2": avg_b2,"Avg BLEU3": avg_b3, "Avg BLEU4": avg_b, "Avg METEOR": avg_m, "Avg Cosine": avg_c
        })

        #plot_metric_histograms(b, b2, b3, m, c, output_dir, model_name=name)


        # save lists
        all_b.append(b)
        all_b2.append(b2)
        all_b3.append(b3)
        all_meteor.append(m)
        all_cosine.append(c)
        model_names.append(name)

        # safer unpacking
        rich_preds = []
        for i, (img_path, pred_caption, bleu4, cosine) in enumerate(preds):
            rich_preds.append((img_path, pred_caption, bleu4, b2[i], b3[i], m[i], cosine))

        sample_outputs.append((name, rich_preds))

    # Now generate plots and tables
    save_metrics_csv(metrics_list, output_dir)

    selected_samples = random.sample(test_data, min(num_samples_to_plot, len(test_data)))
    plot_caption_samples_with_table(selected_samples, sample_outputs, output_dir)

    plot_comparison_metrics(metrics_list, output_dir)
    plot_comparison_line(metrics_list, output_dir)


    all_metrics_dict = {"BLEU2": all_b2,"BLEU3": all_b3,"BLEU4": all_b,"METEOR": all_meteor,"Cosine": all_cosine}
    plot_all_metrics_grid(all_metrics_dict, model_names, output_dir)
