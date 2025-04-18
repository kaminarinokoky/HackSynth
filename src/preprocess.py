import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Select numerical features
    features = ["packet_size", "duration", "bytes_transferred"]
    X = data[features]
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_scaled)
    return X_tensor, scaler