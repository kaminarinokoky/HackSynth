import pandas as pd
import numpy as np


def scrape_cves(limit=5):
    # Mock NVD CVE data (since CVE Details blocks Codespaces)
    mock_cves = [
        ("CVE-2023-12345", "http", 80, "Critical"),
        ("CVE-2022-67890", "ssh", 22, "High"),
        ("CVE-2021-23456", "ftp", 21, "Medium"),
        ("CVE-2020-78901", "mysql", 3306, "High"),
        ("CVE-2019-34567", "smtp", 25, "Low")
    ]
    return mock_cves[:limit]


def generate_vulnerabilities(num_records=1000):
    data = {
        "ip_address": [],
        "port": [],
        "service": [],
        "version": [],
        "vulnerability": [],
        "severity": []
    }
    services = ["http", "ssh", "ftp", "mysql", "smtp"]
    static_cves = [
        ("CVE-2020-1472", "smb", 445, "High"),
        ("CVE-2017-0144", "smb", 445, "Critical"),
        ("CVE-2019-0708", "rdp", 3389, "High"),
        ("None", None, None, "Medium"),
        ("SQL Injection", "http", 80, "High"),
        ("Weak Password", "ssh", 22, "Medium")
    ]
    cve_list = scrape_cves() + static_cves
    cve_indices = list(range(len(cve_list)))
    num_scraped = len(cve_list) - len(static_cves)
    cve_probs = [0.05] * num_scraped + [0.1, 0.1, 0.1, 0.5, 0.05, 0.05]
    cve_probs = cve_probs[:len(cve_list)]  # Truncate to match cve_list
    total_prob = np.sum(cve_probs)
    if total_prob == 0:
        cve_probs = [1 / len(cve_list)] * len(cve_list)
    else:
        cve_probs = np.array(cve_probs) / total_prob

    ports = [22, 80, 443, 21, 3306, 25, 445, 3389]
    for _ in range(num_records):
        ip = f"192.168.1.{np.random.randint(1, 255)}"
        port = np.random.choice(ports)
        service = np.random.choice(services)
        version = f"{np.random.randint(1, 10)}.{np.random.randint(0, 10)}"

        idx = np.random.choice(cve_indices, p=cve_probs)
        vuln, vuln_service, vuln_port, severity = cve_list[idx]

        if vuln_service:
            service = vuln_service
            port = vuln_port or port

        data["ip_address"].append(ip)
        data["port"].append(port)
        data["service"].append(service)
        data["version"].append(version)
        data["vulnerability"].append(vuln)
        data["severity"].append(severity)

    df = pd.DataFrame(data)
    df.to_csv("data/vulnerabilities.csv", index=False)
    print(f"Generated data/vulnerabilities.csv with {len(df)} records")


if __name__ == "__main__":
    generate_vulnerabilities()