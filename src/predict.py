import torch
from model import AnomalyDetector
from preprocess import load_and_preprocess_data

def detect_anomalies():
    # Load data
    file_path = "data/network_traffic.csv"
    X_tensor, scaler = load_and_preprocess_data(file_path)
    
    # Load model
    input_dim = X_tensor.shape[1]
    model = AnomalyDetector(input_dim)
    model.load_state_dict(torch.load("src/model.pth"))
    model.eval()
    
    # Detect anomalies
    with torch.no_grad():
        reconstructions = model(X_tensor)
        mse = torch.mean((reconstructions - X_tensor) ** 2, dim=1)
        threshold = torch.mean(mse) + 2 * torch.std(mse)
        anomalies = mse > threshold
    
    print(f"Detected {anomalies.sum().item()} anomalies")
    # Save results
    anomaly_indices = torch.where(anomalies)[0].numpy()
    print(f"Anomalous record indices: {anomaly_indices}")

if __name__ == "__main__":
    detect_anomalies()