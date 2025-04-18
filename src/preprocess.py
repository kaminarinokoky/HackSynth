import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle


def load_and_preprocess_vuln_data(file_path):
    data = pd.read_csv(file_path)
    features = ["port", "service", "version", "vulnerability"]
    X = data[features]
    ips = data["ip_address"]

    # Define preprocessor
    categorical_features = ["service", "vulnerability"]
    numerical_features = ["port"]
    version_feature = ["version"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("ver", StandardScaler(), version_feature)
        ]
    )

    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)
    X_tensor = torch.FloatTensor(X_transformed.toarray())

    # Save preprocessor for scan.py
    with open("src/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    return X_tensor, preprocessor, ips