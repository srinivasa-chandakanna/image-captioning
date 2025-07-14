import os
import json
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import csv


def load_model_results(selected_dataset, available_models, root_dir="."):
    json_files = {
        display_name: os.path.join(root_dir, "outputs", selected_dataset, f"{model_tag}_training_metadata.json")
        for display_name, model_tag in available_models.items()
    }
    results = {}
    for display_name, file_path in json_files.items():
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                results[display_name] = json.load(f)
            print(f"✅ Loaded: {file_path}")
        else:
            print(f"⚠️ WARNING: File not found: {file_path}")
    return results


def plot_validation_loss(results, selected_dataset, output_dir):
    plt.figure(figsize=(14,6))
    max_len = max(len(data["valid_losses"]) for data in results.values())
    epochs_all = list(range(1, max_len+1))

    for model_name, data in results.items():
        epochs_model = list(range(1, len(data["valid_losses"]) + 1))
        plt.plot(epochs_model, data["valid_losses"], label=model_name, linewidth=2.5)

    plt.title(f"Validation Loss over Epochs ({selected_dataset})")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.xticks(epochs_all)
    plt.xlim(1, max_len)
    plt.grid()
    plt.legend(loc='upper left', bbox_to_anchor=(1.05,1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_loss_comparison.png"), bbox_inches='tight')
    plt.show()


def plot_bleu_scores(results, selected_dataset, output_dir):
    plt.figure(figsize=(14,6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    linestyles = ['-', '--', ':']
    labels = ['BLEU', 'BLEU2', 'BLEU3']
    max_len = max(len(data["bleu_scores"]) for data in results.values())
    epochs_all = list(range(1, max_len+1))

    for i, scores_key in enumerate(['bleu_scores', 'bleu2_scores', 'bleu3_scores']):
        for j, (model_name, data) in enumerate(results.items()):
            epochs_model = list(range(1, len(data[scores_key])+1))
            plt.plot(epochs_model, data[scores_key],
                     label=f"{model_name} {labels[i]}",
                     color=colors[j % len(colors)],
                     linestyle=linestyles[i], linewidth=2.5)

    plt.title(f"BLEU Scores over Epochs ({selected_dataset})")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.xticks(epochs_all)
    plt.xlim(1, max_len)
    plt.grid()
    plt.legend(loc='upper left', bbox_to_anchor=(1.05,1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bleu_scores_comparison.png"), bbox_inches='tight')
    plt.show()

def plot_bleu_scores_separately(results, selected_dataset, output_dir):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    linestyles = ['-', '--', ':']
    score_keys = ['bleu_scores', 'bleu2_scores', 'bleu3_scores']
    labels = ['BLEU', 'BLEU2', 'BLEU3']

    max_len = max(len(data["bleu_scores"]) for data in results.values())
    epochs_all = list(range(1, max_len+1))

    for idx, (score_key, label) in enumerate(zip(score_keys, labels)):
        plt.figure(figsize=(14,6))
        for j, (model_name, data) in enumerate(results.items()):
            epochs_model = list(range(1, len(data[score_key])+1))
            plt.plot(epochs_model, data[score_key],
                     label=model_name,
                     color=colors[j % len(colors)],
                     linestyle=linestyles[0], linewidth=2.5)

        plt.title(f"{label} Score over Epochs ({selected_dataset})")
        plt.xlabel("Epoch")
        plt.ylabel(f"{label} Score")
        plt.xticks(epochs_all)
        plt.xlim(1, max_len)
        plt.grid()
        plt.legend(loc='upper left', bbox_to_anchor=(1.05,1))
        plt.tight_layout()

        # Save each figure
        filename = f"{label.lower()}_scores_comparison.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.show()
        
def plot_cosine_similarity(results, selected_dataset, output_dir):
    plt.figure(figsize=(14,6))
    max_len = max(len(data["cosine_scores"]) for data in results.values())
    epochs_all = list(range(1, max_len+1))

    for model_name, data in results.items():
        epochs_model = list(range(1, len(data["cosine_scores"])+1))
        plt.plot(epochs_model, data["cosine_scores"], label=model_name, linewidth=2.5)

    plt.title(f"Cosine Similarity over Epochs ({selected_dataset})")
    plt.xlabel("Epoch")
    plt.ylabel("Cosine Similarity")
    plt.xticks(epochs_all)
    plt.xlim(1, max_len)
    plt.grid()
    plt.legend(loc='upper left', bbox_to_anchor=(1.05,1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cosine_similarity_comparison.png"), bbox_inches='tight')
    plt.show()


def plot_training_times(results, selected_dataset, output_dir):
    total_times = []
    total_recorded_times = []
    model_names = []

    for model_name, data in results.items():
        sum_train_time = sum(data["time_per_epoch_minutes"])
        total_time_recorded = data["total_training_time_minutes"]
        model_names.append(model_name)
        total_times.append(sum_train_time)
        total_recorded_times.append(total_time_recorded)

    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, total_times, width, label='Sum of Epoch Times', color='tab:blue')
    plt.bar(x + width/2, total_recorded_times, width, label='Total Recorded Time', color='tab:orange')
    plt.ylabel('Time (minutes)')
    plt.title(f'Total Training vs Recorded Time ({selected_dataset})')
    plt.xticks(x, model_names, rotation=15, ha='right')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "total_training_time_comparison.png"))
    plt.show()


def save_summary_table(results, selected_dataset, output_dir):
    table_data = []
    for model_name, data in results.items():
        last_bleu = data["bleu_scores"][-1]
        last_bleu2 = data["bleu2_scores"][-1]
        last_bleu3 = data["bleu3_scores"][-1]
        last_cosine = data["cosine_scores"][-1]
        sum_training_time = sum(data["time_per_epoch_minutes"])
        total_time_recorded = data["total_training_time_minutes"]
        table_data.append([
            model_name,
            round(data["best_loss"], 4),
            data["best_epoch"],
            round(last_bleu, 4),
            round(last_bleu2, 4),
            round(last_bleu3, 4),
            round(last_cosine, 4),
            f'{round(sum_training_time,1)} min',
            f'{round(total_time_recorded,1)} min'
        ])
    table_data.sort(key=lambda x: x[1])

    def bold(text):
        return f"\033[1m{text}\033[0m"

    # Split long headers into 3 lines using \n
    headers = [
        "Model",
        "Best\nVal\nLoss",
        "Best\nEpoch",
        "BLEU",
        "BLEU2",
        "BLEU3",
        "Cosine",
        "Training\nTime\n(sum)",
        "Total\nTime\n(train/val/eval)"
    ]
    bold_headers = [bold(h) for h in headers]

    print("\n" + bold(f"✅ Model Comparison Summary ({selected_dataset}):") + "\n")
    print(tabulate(
        table_data,
        headers=bold_headers,
        tablefmt="pretty",
        colalign=("center",)*len(headers)
    ))

    # Save CSV with normal 1-line headers
    csv_headers = [
        "Model", "Best Val Loss", "Best Epoch",
        "BLEU", "BLEU2", "BLEU3", "Cosine",
        "Training Time (sum epochs)", "Total Time (training/validation/eval)"
    ]
    csv_path = os.path.join(output_dir, "model_comparison_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(table_data)
    print(f"✅ Summary CSV saved to: {csv_path}")



def compare_models_across_metrics(selected_dataset, available_models, root_dir="."):
    output_dir = os.path.join(root_dir, "validation_output", selected_dataset)
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output files will be saved to: {output_dir}")

    results = load_model_results(selected_dataset, available_models, root_dir)
    if not results:
        print("❌ No valid files loaded. Check your paths.")
        return

    plot_validation_loss(results, selected_dataset, output_dir)
    plot_bleu_scores(results, selected_dataset, output_dir)
    plot_bleu_scores_separately(results, selected_dataset, output_dir)
    plot_cosine_similarity(results, selected_dataset, output_dir)
    plot_training_times(results, selected_dataset, output_dir)
    save_summary_table(results, selected_dataset, output_dir)