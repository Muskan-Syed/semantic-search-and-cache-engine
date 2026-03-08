import gdown
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

ARTIFACT_DIR = PROJECT_DIR / "data/artifacts"
PROCESSED_DIR = PROJECT_DIR / "data/processed"


def download_artifacts():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "faiss_index.bin": "1rM0fABxhsj_dR5C3kHH3z1j7w_58sjQZ",
        "document_embeddings.npy": "1ZjzS_UUeILL95c53KQLLf6ChbWgBWxG5",
    }

    for filename, file_id in files.items():
        output = ARTIFACT_DIR / filename

        if not output.exists():
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"Downloading {filename}...")
            gdown.download(url, str(output), quiet=False)


def download_processed_dataset():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset_file = PROCESSED_DIR / "documents_with_clusters.csv"

    if not dataset_file.exists():
        url = "https://drive.google.com/uc?id=1baMCam60GJYscMGEE5xqDYXPFzhD0vJX"
        print("Downloading documents_with_clusters.csv...")
        gdown.download(url, str(dataset_file), quiet=False)
