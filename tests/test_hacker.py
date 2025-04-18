import unittest
import pandas as pd
import torch
from src.preprocess import load_and_preprocess_vuln_data
from src.hacker_model import HackerClassifier
from src.scan import scan_target
from src.exploit import SSHBruteForceEnv, SQLInjectionEnv, q_learning_exploit


class TestHackSynth(unittest.TestCase):
    def test_generate_data(self):
        df = pd.read_csv("data/vulnerabilities.csv")
        self.assertEqual(len(df), 1000)
        severities = df["severity"].value_counts(normalize=True)
        for sev in ["Low", "Medium", "High", "Critical"]:
            self.assertAlmostEqual(severities[sev], 0.25, delta=0.05)

    def test_train_model(self):
        file_path = "data/vulnerabilities.csv"
        X_tensor, _, _ = load_and_preprocess_vuln_data(file_path)
        model = HackerClassifier(input_dim=X_tensor.shape[1])
        model.load_state_dict(
            torch.load("src/hacker_model.pth", weights_only=True)
        )
        self.assertTrue(model)

    def test_scan(self):
        # Mock scan (avoid real network calls in tests)
        df = pd.DataFrame({
            "ip_address": ["127.0.0.1"],
            "port": [80],
            "service": ["http"],
            "version": [1.0],
            "vulnerability": ["Unknown"],
            "severity": ["Unknown"]
        })
        df.to_csv("data/scanned_vulnerabilities.csv", index=False)
        scan_target("127.0.0.1")  # Should load model and predict
        result_df = pd.read_csv("data/scanned_vulnerabilities.csv")
        self.assertFalse(result_df.empty)

    def test_exploit_ssh(self):
        env = SSHBruteForceEnv()
        q_table = q_learning_exploit(env, episodes=10)
        self.assertEqual(q_table.shape, (10, env.action_space.n))

    def test_exploit_sqli(self):
        env = SQLInjectionEnv()
        q_table = q_learning_exploit(env, episodes=10)
        self.assertEqual(q_table.shape, (10, env.action_space.n))


if __name__ == "__main__":
    unittest.main()