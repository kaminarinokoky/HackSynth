import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_network_traffic(num_records=1000):
    start_time = datetime(2025, 4, 18, 12, 0, 0)
    data = {
        "timestamp": [],
        "packet_size": [],
        "duration": [],
        "bytes_transferred": []
    }
    for i in range(num_records):
        data["timestamp"].append((start_time + timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S"))
        is_anomalous = np.random.random() < 0.1  # 10% chance of anomaly
        packet_size = np.random.randint(1500, 5000) if is_anomalous else np.random.randint(64, 1500)
        duration = np.random.exponential(0.1) if is_anomalous else np.random.exponential(0.01)
        bytes_transferred = packet_size * np.random.randint(1, 10)
        data["packet_size"].append(packet_size)
        data["duration"].append(round(duration, 6))
        data["bytes_transferred"].append(bytes_transferred)
    
    df = pd.DataFrame(data)
    df.to_csv("data/network_traffic.csv", index=False)
    print(f"Generated data/network_traffic.csv with {len(df)} records")

if __name__ == "__main__":
    generate_network_traffic()