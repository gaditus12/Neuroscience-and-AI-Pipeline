#!/usr/bin/env python3
import pandas as pd
import umap.umap_ as umap
import matplotlib
matplotlib.use('TkAgg')  # or 'TkAgg' if you're in a GUI environment

# 1) Load your data (adjust the path if needed)
df = pd.read_csv("data/final_sets/all_channels_binary/no_leak/final_final_set/tp7_SPI_norm-Z.csv", index_col=0)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load your CSV

# Extract label and features
labels = df["label"]
X = df.drop(columns=["label", "channel", "session"], errors="ignore")

# Fit LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, labels)

# Plot as a histogram
plt.figure(figsize=(8, 5))
for label in labels.unique():
    sns.kdeplot(X_lda[labels == label].ravel(), fill=True, label=label, alpha=0.5)

plt.xlabel("LDA projection")
plt.title("LDA: Class separation in 1D")
plt.legend(title="Label")
plt.tight_layout()
plt.show()
