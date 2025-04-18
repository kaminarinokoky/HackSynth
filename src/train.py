import torch
import torch.nn as nn
from model import AnomalyDetector
from preprocess import load_and_preprocess_data

def train_model():
    # Load data
    file_path = "data/network_traffic.csv"
    X_tensor, scaler = load_and_preprocess_data(file_path)
    
    # Initialize model
    input_dim = X_tensor.shape[1]
    model = AnomalyDetector(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, X_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    # Save model
    torch.save(model.state_dict(), "src/model.pth")
    print("Model saved to src/model.pth")

if __name__ == "__main__":
    train_model()