import gdown
from pathlib import Path

ARTIFACT_DIR = Path("data/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

files = {
    "faiss_index.bin": "1xfekt8E5ZJLN8GyElQFQvbTFx9VmHg8h",
    "document_embeddings.npy": "17BK0bVAlvw4GNGBzcldQNB9_HnAMnQu4"
}

for filename, file_id in files.items():
    output = ARTIFACT_DIR / filename
    
    if not output.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        gdown.download(url, str(output), quiet=False)
