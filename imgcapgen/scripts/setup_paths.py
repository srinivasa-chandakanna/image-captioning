# imgcapgen/scripts/setup_paths.py
import os
import sys
from pathlib import Path
import kagglehub


def add_project_root_to_path():
    """
    Ensures the repo root is on sys.path so imports work.
    """
    root_path = Path(__file__).resolve().parents[2]  # from scripts → imgcapgen → repo root
    if str(root_path) not in sys.path:
        sys.path.append(str(root_path))
        


def setup_paths(cfg):
    """
    Sets up all paths dynamically based on loaded cfg.
    Places artifacts and outputs in central folders by dataset.
    """
    try:
        import google.colab
        ON_COLAB = True
    except ImportError:
        ON_COLAB = False
    
    
    if ON_COLAB:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        script_path = Path(os.getcwd()).resolve()
        if "/content/drive/" not in str(script_path):
            raise RuntimeError("Script is not running from inside Google Drive.")
        root = script_path
        print("🚀 Running on Google Colab inside Google Drive")
    else:
        root = Path(os.getcwd()).resolve()
        print("💻 Running on Local Machine:", root)
    
    
    
    print(f"[INFO] Root directory detected: {root}")
    
    
    # Load dataset + source from cfg
    dataset = cfg.dataset.lower()
    data_source = cfg.data_source.lower()
    dataset_url = Path(cfg.dataset_url)
    yaml_image_path = Path(cfg.image_path)
    yaml_captions_path = Path(cfg.captions_path)
    

    
    if data_source == "google_drive":
        image_path = root / dataset_url / yaml_image_path
        captions_path = root / dataset_url / yaml_captions_path

    elif data_source == "kaggle":
        dataset_source = str(dataset_url).replace("\\", "/")
        download_path = kagglehub.dataset_download(dataset_source, force_download=True)
        print(f"[INFO] Kaggle dataset downloaded to: {download_path}")
        download_path = Path(download_path)
        image_path = download_path / dataset / yaml_image_path
        captions_path = download_path / dataset / yaml_captions_path

    else:
        raise ValueError("Invalid data_source. Use 'google_drive' or 'kaggle'.")   
    
    
    # Centralized artifacts & outputs directories with dataset subfolders
    artifact_dir = root / cfg.artifact_dir / dataset
    output_dir = root / cfg.output_dir / dataset
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update cfg
    # Update cfg
    cfg.image_path = str(image_path)
    cfg.captions_path = str(captions_path)
    cfg.artifact_dir = str(artifact_dir)
    cfg.output_dir = str(output_dir)

    print("\n✅ Paths configured:")
    print(f"📂 Image path:    {cfg.image_path}")
    print(f"📝 Captions path: {cfg.captions_path}")
    print(f"💾 Artifacts dir: {cfg.artifact_dir}")
    print(f"💾 Output dir:    {cfg.output_dir}")
