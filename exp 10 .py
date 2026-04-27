import os
import numpy as np
from sklearn.mixture import GaussianMixture

# --- Fix for joblib warning on Windows ---
# Set logical core count explicitly to avoid warning
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())

# Generate synthetic dataset
np.random.seed(42)
X = np.concatenate([
    np.random.normal(0, 1, 100),
    np.random.normal(5, 1, 100)
]).reshape(-1, 1)

# Apply EM using Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, max_iter=100, random_state=42)
gmm.fit(X)

# Predict cluster labels
labels = gmm.predict(X)

# Results
print("Means of clusters:", gmm.means_.flatten())
print("Covariances of clusters:", gmm.covariances_.flatten())
print("Cluster assignments (first 20):", labels[:20])
