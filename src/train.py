import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from preprocess import load_and_preprocess_vuln_data
from hacker_model import HackerClassifier


def train_model():
    file_path = "data/vulnerabilities.csv"
    data = pd.read_csv(file_path)

    # Clean severity data
    valid_severities = ["Low", "Medium", "High", "Critical"]
    data["severity"] = data["severity"].fillna("Medium")
    data["severity"] = data["severity"].apply(
        lambda x: x if x in valid_severities else "Medium"
    )
    data = data[data["severity"].isin(valid_severities)].copy()

    if data.empty:
        print("No valid severity data in vulnerabilities.csv. Run generate_data.py.")
        return

    # Preprocess
    X_tensor, preprocessor, _ = load_and_preprocess_vuln_data(file_path)
    X_tensor = X_tensor[:len(data)]  # Match cleaned data

    # Create labels (Low=0, Medium=1, High=2, Critical=3)
    severity_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    y = data["severity"].map(severity_map).values
    y_tensor = torch.LongTensor(y)

    # Validate labels
    if y_tensor.min() < 0 or y_tensor.max() > 3:
        raise ValueError(f"Invalid labels: {y_tensor.tolist()}. Expected [0, 3].")

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_tensor, y_tensor, test_size=0.2, stratify=y_tensor, random_state=42
    )

    # Datasets and DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model, loss, optimizer
    model = HackerClassifier(input_dim=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Increased lr

    # Early stopping
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    epochs = 100

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        model.train()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Train Loss: {avg_train_loss:.6f}, "
                f"Val Loss: {avg_val_loss:.6f}"
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "src/hacker_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print("Saved trained model to src/hacker_model.pth")


if __name__ == "__main__":
    train_model()