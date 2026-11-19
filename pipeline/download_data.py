import os
import requests

# URL and destination path
URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "yellow_tripdata_2025-01.parquet")

def download_taxi_data():
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(OUTPUT_FILE):
        print(f"File already exists at {OUTPUT_FILE}. Skipping download.")
        return

    print(f"Downloading January 2025 Yellow Taxi data from {URL}...")
    response = requests.get(URL, stream=True)
    
    if response.status_code == 200:
        with open(OUTPUT_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded dataset to {OUTPUT_FILE}!")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

if __name__ == "__main__":
    download_taxi_data()