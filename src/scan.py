import nmap
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
from src.hacker_model import HackerClassifier


def scan_target(target="5.144.129.200"):
    try:
        print(
            "BIG FUCKING WARNING: Only scan systems you OWN or have EXPLICIT "
            "WRITTEN PERMISSION to test! Unauthorized scanning is ILLEGAL."
        )
        nm = nmap.PortScanner()
        nm.scan(target, arguments="-sV")

        data = {
            "ip_address": [],
            "port": [],
            "service": [],
            "version": [],
            "vulnerability": [],
            "severity": []
        }

        for host in nm.all_hosts():
            for proto in nm[host].all_protocols():
                for port in nm[host][proto].keys():
                    service = nm[host][proto][port]["name"]
                    version = nm[host][proto][port].get("version", "1.0")
                    try:
                        version = float(version.split()[0])
                    except (ValueError, IndexError):
                        version = 1.0
                    data["ip_address"].append(host)
                    data["port"].append(port)
                    data["service"].append(service)
                    data["version"].append(version)
                    data["vulnerability"].append("Unknown")
                    data["severity"].append("Unknown")

        if not data["ip_address"]:
            print(f"No open ports found on {target}. Is it up?")
            return

        df = pd.DataFrame(data)
        df.to_csv("data/scanned_vulnerabilities.csv", index=False)

        try:
            with open("src/preprocessor.pkl", "rb") as f:
                preprocessor = pickle.load(f)
        except FileNotFoundError:
            print("No preprocessor found. Run src/train.py first.")
            return

        features = ["port", "service", "version", "vulnerability"]
        X = df[features]
        try:
            X_transformed = preprocessor.transform(X)
        except Exception as e:
            print(f"Preprocessing failed: {e}. Check nmap output or preprocessor.")
            return

        X_tensor = torch.FloatTensor(X_transformed.toarray())
        ips = df["ip_address"]

        model = HackerClassifier(input_dim=X_tensor.shape[1])
        try:
            model.load_state_dict(
                torch.load("src/hacker_model.pth", weights_only=True)
            )
            print("Loaded trained model from src/hacker_model.pth")
        except FileNotFoundError:
            print("No trained model found. Run src/train.py first.")
            return

        model.eval()
        with torch.no_grad():
            preds = model(X_tensor)
            severities = ["Low", "Medium", "High", "Critical"]
            pred_labels = [severities[p.argmax().item()] for p in preds]

        print("Scan Results:")
        for ip, port, sev in zip(ips, df["port"], pred_labels):
            print(f"IP: {ip}, Port: {port}, Predicted Severity: {sev}")

        severity_counts = pd.Series(pred_labels).value_counts()
        plt.figure(figsize=(8, 6))
        severity_counts.plot(kind="bar", color="red")
        plt.title("Vulnerability Severity Distribution")
        plt.xlabel("Severity")
        plt.ylabel("Count")
        plt.savefig("vuln_plot.png")
        plt.close()
        print("Saved severity plot to vuln_plot.png")

    except Exception as e:
        print(f"Scan failed: {e}. Check target IP or nmap installation.")


if __name__ == "__main__":
    scan_target()