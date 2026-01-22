import os
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok=True)

# --- 1. Download the first dataset ---
print("Downloading dataset: chethuhn/network-intrusion-dataset ...")
network_intrusion_path = kagglehub.dataset_download(
    "chethuhn/network-intrusion-dataset", 
    path=data_dir
)
print("Dataset saved at:", network_intrusion_path)

# --- 2. Download the second dataset ---
print("\nDownloading dataset: arnobbhowmik/ton-iot-network-dataset ...")
ton_iot_path = kagglehub.dataset_download(
    "arnobbhowmik/ton-iot-network-dataset", 
    path=data_dir
)
print("Dataset saved at:", ton_iot_path)

csv_files = [f for f in os.listdir(ton_iot_path) if f.endswith(".csv")]

if csv_files:
    file_path = os.path.join(ton_iot_path, csv_files[0])
    
    df = pd.read_csv(file_path)
    
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
else:
    print("\nNo CSV files found in the dataset:", ton_iot_path)
