import os
import tarfile
from pathlib import Path

import requests


def download_file(url: str, destination: Path):
    print(f"Downloading {url} to {destination}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, "wb") as f:
            total_downloaded = 0
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    total_downloaded += len(chunk)
                    if total_downloaded >= 10485760:  # 10 MB
                        print(".", end="", flush=True)
                        total_downloaded = 0
        print("\nDownload complete.")
    else:
        print(f"Download failed with status code {response.status_code}")


def download_model(model_data_dir: str, whisper_model_name: str):
    model_folder = Path(model_data_dir, whisper_model_name)

    model_folder.mkdir(parents=True)

    # config.json, model.bin, tokenizer.json, vocabulary.json
    download_path = Path(model_data_dir, f"{whisper_model_name}.tar")
    download_file(
        "https://www.dropbox.com/scl/fi/tc4d2xuf23ra99mvwp4ms/WhisperCHsmall.tar?dl=1&rlkey=ifx4evisyh09d7yistwlo4kz5",
        download_path,
    )

    with tarfile.open(download_path, "r") as tar_file:
        tar_file.extractall(model_folder)

    # delete downloaded file after extraction
    os.remove(download_path)
