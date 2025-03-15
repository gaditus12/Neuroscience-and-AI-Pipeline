import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ---- CONFIGURATION ----
FEATURES_FILE = "data/merged_features/baseline_imagery_instruction_merge/merged_features_threezones.csv"
TOP_N_LABELS = 2  # Set to 2 for most diverse duo, 3 for trio
N_FEATURES_TO_SELECT = 15  # Number of top features to select

# Approach option
CHANNEL_APPROACH = "pooled"  # Options: "pooled", "separate", "features"

# ---- LOAD DATA ----
df = pd.read_csv(FEATURES_FILE)

# Ensure we have channel column
if 'channel' not in df.columns:
    # If your data format has channel as the first column (as in your example)
    df['channel'] = df.iloc[:, 0]  # Adjust if needed

# Get feature columns (everything except metadata)
feature_columns = [col for col in df.columns if col not in ["label", "channel", "session"]]

# Get unique labels and channels
unique_labels = df["label"].unique()
unique_channels = df['channel'].unique() if 'channel' in df.columns else ["unknown"]

print(f"Dataset has {len(df)} rows with {len(feature_columns)} features")
print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
print(f"Found {len(unique_channels)} unique channels: {unique_channels}")

# Print sample counts
print("\nSample counts per label:")
print(df.groupby("label").size())

if 'channel' in df.columns:
    print("\nSample counts per label and channel:")
    print(df.groupby(["label", "channel"]).size())

# ---- HANDLE CHANNELS BASED ON APPROACH ----
if CHANNEL_APPROACH == "pooled":
    # Treat each row as an independent sample, keeping channel as a feature
    # (This is the approach to use given your small sample size)
    print("\nUsing POOLED approach: Treating each channel reading as an independent sample")
    X = df[feature_columns]
    y = df["label"]

elif CHANNEL_APPROACH == "separate":
    # Analyze each channel separately (not recommended for small datasets)
    print("\nUsing SEPARATE approach: Analyzing each channel independently")
    # We'll implement this later in the channel analysis section
    X = df[feature_columns]
    y = df["label"]

else:  # "features" approach
    # Combine channels as additional features for each sample
    # This approach only works if your data is structured with multiple rows per actual sample
    # (one row per channel for the same recording)
    print("\nUsing FEATURES approach: Combining channels as additional features")
    # Assuming session identifies unique recordings
    if 'session' not in df.columns:
        print("Error: This approach requires a 'session' column to identify unique recordings")
        CHANNEL_APPROACH = "pooled"
        X = df[feature_columns]
        y = df["label"]

# Group by session and label, creating wide-format data with channels as features
# This is complex and assumes data structure we don't have complete info on,
# so we'll stick with the pooled approach for now

# ---- PREPROCESSING ----
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

# Add this code after the preprocessing section and before the current feature selection section

# ---- ENHANCED FEATURE SELECTION ----
print("\nUsing enhanced feature selection with multiple scoring methods...")
# Load feature importance rankings from file
try:
    # TODO uncomment this line (for now we are just checking the three zones)
    # importance_df = pd.read_csv("data/merged_features/merge_run_1741541389/imagery_task_feature_ranking.csv")  # Update with your actual file path
    print(f"Loaded feature importance rankings for {len(importance_df)} features")

    # Sort by the ensemble score (which combines multiple methods)
    importance_df = importance_df.sort_values(by='ensemble_score', ascending=False)

    # Select the top N features based on ensemble score
    top_features = importance_df.head(N_FEATURES_TO_SELECT)['feature'].tolist()

    print("Selected top features based on ensemble scoring:")
    for i, feature in enumerate(top_features):
        if feature in feature_columns:
            print(
                f"  {i + 1}. {feature} (ensemble score: {importance_df[importance_df['feature'] == feature]['ensemble_score'].values[0]:.4f})")
        else:
            print(f"  {i + 1}. {feature} - WARNING: Not found in current dataset")

    # Filter X_scaled to only include top features that exist in our dataset
    available_top_features = [f for f in top_features if f in feature_columns]
    if len(available_top_features) < N_FEATURES_TO_SELECT:
        print(
            f"Warning: Only {len(available_top_features)} of the top {N_FEATURES_TO_SELECT} features are available in the current dataset")

        # Supplement with ANOVA selection for any missing features
        missing_count = N_FEATURES_TO_SELECT - len(available_top_features)
        print(f"Supplementing with {missing_count} additional features using ANOVA F-test")

        # Get F-values for all features
        selector = SelectKBest(f_classif, k='all')
        selector.fit(X_scaled, y)
        f_values = selector.scores_

        # Create a dataframe of features with their F-values
        feature_f_scores = pd.DataFrame({
            'feature': feature_columns,
            'f_score': f_values
        })

        # Remove features that are already selected
        feature_f_scores = feature_f_scores[~feature_f_scores['feature'].isin(available_top_features)]

        # Sort by F-value and take the top missing_count features
        additional_features = feature_f_scores.sort_values('f_score', ascending=False).head(missing_count)[
            'feature'].tolist()

        # Add these to our selected features
        available_top_features.extend(additional_features)

        print("Additional features selected via ANOVA:")
        for i, feature in enumerate(additional_features):
            f_score = feature_f_scores[feature_f_scores['feature'] == feature]['f_score'].values[0]
            print(f"  {len(top_features) + i + 1}. {feature} (F-score: {f_score:.4f})")

    # Select columns from the dataset
    X_selected = X_scaled_df[available_top_features].values
    selected_features = available_top_features

except Exception as e:
    print(f"Could not load feature importance file: {str(e)}")
    print("Falling back to standard ANOVA F-test feature selection")

    # Standard SelectKBest from the original code
    selector = SelectKBest(f_classif, k=N_FEATURES_TO_SELECT)
    X_selected = selector.fit_transform(X_scaled, y)

    # Get selected feature names for interpretation
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_columns[i] for i in selected_indices]
    print("Selected features:")
    for i, feature in enumerate(selected_features):
        print(f"  {i + 1}. {feature}")


# If the external file was loaded and processed successfully, wrap our feature list into a dummy selector.
if 'selected_features' not in locals():
    # We're in the fallback, so selector is already a SelectKBest instance.
    pass
else:
    # Create a dummy selector so that downstream calls to selector.transform() work.
    class DummySelector:
        def __init__(self, feature_columns, selected_features):
            self.feature_columns = feature_columns
            self.selected_features = selected_features
        def transform(self, X):
            # Subset the DataFrame/array to the indices corresponding to the selected features.
            # We assume that the columns of X correspond to feature_columns.
            indices = [self.feature_columns.index(f) for f in self.selected_features]
            return X[:, indices]
        def get_support(self, indices=False):
            mask = [f in self.selected_features for f in self.feature_columns]
            if indices:
                return [i for i, m in enumerate(mask) if m]
            return mask

    selector = DummySelector(feature_columns, selected_features)


# # Get selected feature names for interpretation
# selected_indices = selector.get_support(indices=True)
# selected_features = [feature_columns[i] for i in selected_indices]
# print("Selected features:")
# for i, feature in enumerate(selected_features):
#     print(f"  {i + 1}. {feature}")

# ---- FIND MOST SEPARABLE LABEL PAIRS ----
print("\nFinding most separable label combinations...")
separability_scores = {}
detailed_results = {}

# Using Leave-One-Out cross-validation for small dataset
loo = LeaveOneOut()

# Iterate over all possible label pairs (or triplets)
for label_combo in combinations(unique_labels, TOP_N_LABELS):
    df_subset = df[df["label"].isin(label_combo)]
    X_subset = df_subset[feature_columns]

    # Standardize subset
    X_subset_scaled = scaler.transform(X_subset)

    # Apply feature selection to subset
    X_subset_selected = selector.transform(X_subset_scaled)

    y_subset = df_subset["label"]

    # Print sample count for this combination
    combo_sample_count = df_subset.groupby("label").size()
    print(f"Label combination {label_combo}: {len(df_subset)} samples")
    print(f"  Per label: {combo_sample_count.to_dict()}")

    # Try both Random Forest and SVM
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
    }

    model_scores = {}
    for name, model in models.items():
        # Use Leave-One-Out CV for more reliable results on small dataset
        cv_scores = cross_val_score(model, X_subset_selected, y_subset, cv=loo, scoring='accuracy')
        model_scores[name] = np.mean(cv_scores)
        print(f"  {name} accuracy: {model_scores[name]:.4f}")

    # Select best model score
    best_model = max(model_scores, key=model_scores.get)
    best_score = model_scores[best_model]

    separability_scores[label_combo] = best_score
    detailed_results[label_combo] = {
        'best_model': best_model,
        'scores': model_scores,
        'n_samples': len(y_subset),
        'sample_counts': combo_sample_count.to_dict()
    }

# Sort and get the most separable labels
best_labels = max(separability_scores, key=separability_scores.get)
print(f"\nThe {TOP_N_LABELS} most diverse labels are: {best_labels}")
print(f"Separation Score: {separability_scores[best_labels]:.4f}")
print(f"Best model: {detailed_results[best_labels]['best_model']}")
print(f"Sample counts: {detailed_results[best_labels]['n_samples']} total samples")

# ---- VISUALIZATIONS ----
# Create a better color palette
palette = sns.color_palette("colorblind", n_colors=len(unique_labels))
color_dict = {label: palette[i] for i, label in enumerate(unique_labels)}

# Create marker dictionary for channels
markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'P', 'X']
marker_dict = {channel: markers[i % len(markers)] for i, channel in enumerate(unique_channels)}

# 1. PCA VISUALIZATION
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["label"] = y
if 'channel' in df.columns:
    df_pca["channel"] = df["channel"]

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(12, 10))
# Create scatter plot with different markers for channels
for label in unique_labels:
    label_data = df_pca[df_pca['label'] == label]

    if 'channel' in df.columns:
        for channel in unique_channels:
            channel_data = label_data[label_data['channel'] == channel]
            if len(channel_data) > 0:
                plt.scatter(
                    channel_data["PC1"],
                    channel_data["PC2"],
                    s=100,
                    c=[color_dict[label]],
                    marker=marker_dict[channel],
                    alpha=0.8,
                    edgecolor='w',
                    linewidth=0.5,
                    label=f"{label} ({channel})"
                )
    else:
        plt.scatter(
            label_data["PC1"],
            label_data["PC2"],
            s=100,
            c=[color_dict[label]],
            alpha=0.8,
            edgecolor='w',
            linewidth=0.5,
            label=label
        )

# Draw ellipses around each label's data points (ignoring channels for clarity)
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of x and y.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from the square root of the variance
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # Calculating the standard deviation of y from the square root of the variance
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# Add confidence ellipses for each label
for label in unique_labels:
    label_data = df_pca[df_pca['label'] == label]
    if len(label_data) >= 3:  # Need at least 3 points for covariance
        confidence_ellipse(
            label_data["PC1"],
            label_data["PC2"],
            plt.gca(),
            n_std=2.0,
            edgecolor=color_dict[label],
            linewidth=2,
            alpha=0.5
        )

# Add centroids for each label
for label in unique_labels:
    label_data = df_pca[df_pca['label'] == label]
    centroid_x = label_data['PC1'].mean()
    centroid_y = label_data['PC2'].mean()
    plt.scatter(
        centroid_x, centroid_y,
        s=200,
        c=[color_dict[label]],
        marker='X',
        edgecolor='black',
        linewidth=1.5,
        alpha=1.0
    )
    plt.annotate(
        f"{label}",
        (centroid_x, centroid_y),
        fontsize=12,
        fontweight='bold',
        ha='center',
        va='bottom',
        xytext=(0, 10),
        textcoords='offset points',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
    )

# Add title and labels
plt.title("PCA Visualization of EEG Features\nChannel markers show distribution of readings", fontsize=16, pad=20)
plt.xlabel(f"PC1 ({explained_variance[0]:.2f}% Variance)", fontsize=12)
plt.ylabel(f"PC2 ({explained_variance[1]:.2f}% Variance)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
# Handle the legend to avoid duplicates
by_label = {}
handles, labels = plt.gca().get_legend_handles_labels()
for h, l in zip(handles, labels):
    label_part = l.split(' ')[0]  # Get just the label part, not the channel
    if label_part not in by_label:
        by_label[label_part] = h

# Add legend for labels only
plt.legend(by_label.values(), by_label.keys(),
           title="Labels", title_fontsize=12, fontsize=10,
           loc='best', frameon=True, framealpha=0.95)

# Add a second legend for channel markers if needed
if 'channel' in df.columns:
    marker_handles = [plt.Line2D([0], [0], marker=marker_dict[ch], color='gray',
                                 linestyle='None', markersize=10)
                      for ch in unique_channels]
    marker_labels = [f"Channel: {ch}" for ch in unique_channels]

    # Create the second legend outside the plot
    plt.figlegend(marker_handles, marker_labels,
                  loc='lower center', ncol=len(unique_channels),
                  bbox_to_anchor=(0.5, 0), fontsize=10, frameon=True)
    plt.subplots_adjust(bottom=0.15)  # Make room for channel legend

# Add text about best labels
plt.figtext(
    0.5, 0.01,
    f"Most Separable Labels: {', '.join(best_labels)}\n"
    f"Separation Score: {separability_scores[best_labels]:.4f} using {detailed_results[best_labels]['best_model']}",
    ha="center", fontsize=14,
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8)
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make room for the annotation at the bottom

# Save the visualization
output_file = f"{CHANNEL_APPROACH}_eeg_pooled_visualization_top{TOP_N_LABELS}_labels_top{N_FEATURES_TO_SELECT}_features.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Enhanced EEG visualization saved as '{output_file}'")

# ---- FEATURE IMPORTANCE VISUALIZATION ----
# Create a feature importance plot for the best labelsCHANNEL_APPROACH
plt.figure(figsize=(14, 10))

# Get the features that best distinguish between the best labels
df_best = df[df["label"].isin(best_labels)]
X_best = df_best[feature_columns]
y_best = df_best["label"]

# Standardize
X_best_scaled = scaler.transform(X_best)

# Get ANOVA F-values for ranking
f_values, p_values = f_classif(X_best_scaled, y_best)
feature_scores = pd.DataFrame({
    'Feature': feature_columns,
    'F_Score': f_values,
    'P_Value': p_values,
    'Log10_F': np.log10(f_values + 1)  # Log transform for better visualization
})
feature_scores = feature_scores.sort_values('F_Score', ascending=False)

# Color significant features differently
feature_scores['Significant'] = feature_scores['P_Value'] < 0.05
feature_scores['Color'] = feature_scores['Significant'].map({True: 'darkblue', False: 'lightblue'})

# Plot all features
plt.figure(figsize=(14, 10))
bars = plt.barh(feature_scores['Feature'], feature_scores['Log10_F'], color=feature_scores['Color'])
plt.title(f'Feature Importance for Distinguishing Between {best_labels}', fontsize=16)
plt.xlabel('Log10(F-Score+1) - Higher Values = More Discriminative', fontsize=12)
plt.ylabel('EEG Feature', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add a line for significance threshold
sig_features = feature_scores[feature_scores['Significant']]
if not sig_features.empty:
    min_sig_log_f = np.log10(sig_features['F_Score'].min() + 1)
    plt.axvline(x=min_sig_log_f, color='red', linestyle='--', alpha=0.7)
    plt.text(min_sig_log_f + 0.1, 1, 'Significance Threshold (p<0.05)',
             rotation=90, color='red', verticalalignment='bottom')

# Add annotations with actual F-scores and p-values
for i, (_, row) in enumerate(feature_scores.iterrows()):
    plt.text(
        row['Log10_F'] + 0.1,
        i,
        f"F={row['F_Score']:.2f}, p={row['P_Value']:.4f}",
        va='center',
        fontsize=9
    )

plt.tight_layout()

# Save the feature importance plot
importance_file = f"{CHANNEL_APPROACH}_feature_importance_top{TOP_N_LABELS}_labels_top{N_FEATURES_TO_SELECT}_features.png"
plt.savefig(importance_file, dpi=300, bbox_inches='tight')
print(f"Feature importance visualization saved as '{importance_file}'")

# ---- BONUS: ANALYZE CHANNEL INFORMATION IF AVAILABLE ----
if 'channel' in df.columns and CHANNEL_APPROACH == "pooled":
    plt.figure(figsize=(12, 8))

    # Get the channel distribution for each label
    label_channel_counts = pd.crosstab(df['label'], df['channel'])

    # Normalize to percentage
    label_channel_pct = label_channel_counts.div(label_channel_counts.sum(axis=1), axis=0) * 100

    # Plot as heatmap
    plt.figure(figsize=(len(unique_channels) * 1.5, len(unique_labels) * 1.2))
    sns.heatmap(label_channel_pct, annot=label_channel_counts, fmt="d", cmap="YlGnBu",
                cbar_kws={'label': 'Sample Percentage (%)'})
    plt.title('Sample Distribution by Label and Channel', fontsize=16)
    plt.ylabel('Label', fontsize=12)
    plt.xlabel('Channel', fontsize=12)
    plt.tight_layout()

    # Save the channel distribution plot
    channel_file = "channel_distribution.png"
    plt.savefig(channel_file, dpi=300, bbox_inches='tight')
    print(f"Channel distribution saved as '{channel_file}'")