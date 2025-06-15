import pandas as pd
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import os

# -----------------------------
# Parameters
# -----------------------------
GENE_FILTER_TPM_THRESHOLD = 1.0
DEFAULT_N_INIT = 10
DEFAULT_MIN_SAMPLES = 5
DEFAULT_EPS_RANGE = np.arange(0.5, 5.5, 0.5)

# -----------------------------
# Data Preprocessing
# -----------------------------
def load_and_preprocess(file_path):
    """Loads and preprocesses gene expression data from a file.

    Args:
        file_path (str): Path to the gene expression data file (GCT format).

    Returns:
        tuple: A tuple containing the preprocessed data (X_z.T) and the DataFrame (df).
    """
    print("[INFO] Loading and preprocessing expression data...")

    if not file_path.endswith((".gct", ".gct.gz")):
        raise ValueError("Input file must be a .gct or .gct.gz file")

    df = pd.read_csv(file_path, sep='\t', skiprows=2)
    df = df.set_index('Name').drop(columns=['Description'])
    df = df[df.mean(axis=1) > GENE_FILTER_TPM_THRESHOLD]
    X = df.values.astype(float)
    X_z = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    return X_z.T, df

# -----------------------------
# PCA
# -----------------------------
def apply_pca(X, n_components=50):
    """Applies Principal Component Analysis (PCA) to the data.

    Args:
        X (np.ndarray): The input data.
        n_components (int): The number of PCA components to retain.

    Returns:
        np.ndarray: The PCA-transformed data.
    """
    print("[INFO] Applying PCA...")
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X)
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid()
    plt.savefig("figures/pca_variance.png")
    plt.show()
    return X_pca

# -----------------------------
# Optimal K Selection
# -----------------------------
def plot_elbow_silhouette(X, max_clusters=10):
    """Determines the optimal number of clusters (k) using the elbow method and silhouette analysis.

    Args:
        X (np.ndarray): The input data.
        max_clusters (int): The maximum number of clusters to consider.

    Returns:
        int: The optimal number of clusters based on silhouette analysis.
    """
    distortions, silhouettes = [], []
    Ks = range(2, max_clusters + 1)
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=0, n_init=DEFAULT_N_INIT).fit(X)
        distortions.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(Ks, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(Ks, silhouettes, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.grid()
    plt.tight_layout()
    plt.savefig("figures/k_selection.png")
    plt.show()
    best_k = Ks[int(np.argmax(silhouettes))]
    print(f"[INFO] Optimal k by silhouette: {best_k}")
    return best_k

# -----------------------------
# DBSCAN Tuning
# -----------------------------
def tune_dbscan(X, eps_range=DEFAULT_EPS_RANGE):
    """Tunes the DBSCAN epsilon parameter using silhouette analysis.

    Args:
        X (np.ndarray): The input data.
        eps_range (np.ndarray): The range of epsilon values to consider.

    Returns:
        np.ndarray: The cluster labels assigned by DBSCAN with the best epsilon.
    """
    best_score, best_eps = -1, None
    for eps in eps_range:
        labels = DBSCAN(eps=eps, min_samples=DEFAULT_MIN_SAMPLES).fit_predict(X)
        mask = labels != -1
        if mask.sum() > 1:  # Only evaluate silhouette if clusters are found
            score = silhouette_score(X[mask], labels[mask])
            print(f"[DBSCAN] eps={eps:.1f} → Silhouette: {score:.3f}")
            if score > best_score:
                best_score, best_eps = score, eps
        else:
            print(f"[DBSCAN] eps={eps:.1f} → Not enough core samples")
    if best_eps is not None:
        print(f"[INFO] Best DBSCAN eps: {best_eps:.1f} (Silhouette: {best_score:.3f})")
        return DBSCAN(eps=best_eps, min_samples=DEFAULT_MIN_SAMPLES).fit_predict(X)
    print("[WARNING] No valid DBSCAN eps found")
    return np.full(X.shape[0], -1)

# -----------------------------
# Clustering Visualization
# -----------------------------
def plot_clusters(X, labels, title):
    """Plots the clustering results in 2D and 3D.

    Args:
        X (np.ndarray): The input data (PCA-transformed).
        labels (np.ndarray): The cluster labels.
        title (str): The title of the plot.
    """
    unique = np.unique(labels)
    plt.figure(figsize=(6, 5))
    for lab in unique:
        mask = labels == lab
        plt.scatter(X[mask, 0], X[mask, 1], label=str(lab), s=20, alpha=0.7)
    plt.title(f"{title} - 2D")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"figures/{title}_2D.png")
    plt.show()

    if X.shape[1] >= 3:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        for lab in unique:
            mask = labels == lab
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], label=str(lab), s=20, alpha=0.7)
        ax.set_title(f"{title} - 3D")
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.legend(bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(f"figures/{title}_3D.png")
        plt.show()

# -----------------------------
# Full Pipeline
# -----------------------------
def perform_clustering(X, k, df, meta_file):
    """Performs clustering using KMeans, GMM, and DBSCAN, and evaluates the results.

    Args:
        X (np.ndarray): The input data (PCA-transformed).
        k (int): The number of clusters for KMeans and GMM.
        df (pd.DataFrame): The original DataFrame.
        meta_file (str): Path to the metadata file.
    """
    print("[INFO] Running clustering algorithms...")

    # KMeans
    k_labels = KMeans(n_clusters=k, random_state=0, n_init=DEFAULT_N_INIT).fit_predict(X)
    print('K-Means:', silhouette_score(X, k_labels), calinski_harabasz_score(X, k_labels), davies_bouldin_score(X, k_labels))
    plot_clusters(X, k_labels, 'K-Means')

    # GMM
    g_labels = GaussianMixture(n_components=k, random_state=0).fit_predict(X)
    print('GMM:', silhouette_score(X, g_labels), calinski_harabasz_score(X, g_labels), davies_bouldin_score(X, g_labels))
    plot_clusters(X, g_labels, 'GMM')

    # DBSCAN
    d_labels = tune_dbscan(X)
    mask = d_labels != -1
    if mask.sum() > 1:
        print('DBSCAN:', silhouette_score(X[mask], d_labels[mask]), calinski_harabasz_score(X[mask], d_labels[mask]), davies_bouldin_score(X[mask], d_labels[mask]))
    plot_clusters(X, d_labels, 'DBSCAN')

    # External validation
    try:
        meta = pd.read_csv(meta_file, sep='\t')
        true = meta.set_index('SAMPID').loc[df.columns]['SMTSD'].values
        print('ARI:', adjusted_rand_score(true, k_labels), 'NMI:', normalized_mutual_info_score(true, k_labels))

        emb = TSNE(n_components=2, random_state=0).fit_transform(X)
        plt.figure(figsize=(6, 5))
        plt.scatter(emb[:, 0], emb[:, 1], c=k_labels, s=20)
        plt.title('t-SNE Visualization')
        plt.savefig("figures/tsne.png")
        plt.show()

        cm = pd.crosstab(pd.Series(true, name='Tissue'), pd.Series(k_labels, name='Cluster'))
        plt.figure(figsize=(6, 5))
        plt.imshow(cm.values, cmap='YlGnBu', aspect='auto')
        plt.colorbar(label='Count')
        plt.xticks(range(cm.shape[1]), cm.columns, rotation=90)
        plt.yticks(range(cm.shape[0]), cm.index)
        plt.title('Confusion Matrix')
        plt.xlabel('Cluster')
        plt.ylabel('Tissue')
        plt.tight_layout()
        plt.savefig("figures/confusion_matrix.png")
        plt.show()
    except FileNotFoundError:
        print("[ERROR] Metadata file not found.")
    except KeyError as ke:
        print(f"[ERROR] Column mismatch in metadata: {ke}")
    except pd.errors.EmptyDataError:
        print("[ERROR] Metadata file is empty.")
    except Exception as e:
        print(f"[ERROR] Unknown error during metadata processing: {e}")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, help="Path to GTEx expression matrix (.gct.gz)")
    parser.add_argument("--meta", required=True, help="Path to GTEx metadata file")
    parser.add_argument("--pca", type=int, default=50, help="Number of PCA components (default: 50)")
    parser.add_argument("--eps_range_start", type=float, default=0.5, help="Start value for DBSCAN eps range (default: 0.5)")
    parser.add_argument("--eps_range_end", type=float, default=5.5, help="End value for DBSCAN eps range (default: 5.5)")
    parser.add_argument("--eps_range_step", type=float, default=0.5, help="Step value for DBSCAN eps range (default: 0.5)")
    args = parser.parse_args()

    try:
        X, df = load_and_preprocess(args.exp)
        X_pca = apply_pca(X, n_components=args.pca)
        k = plot_elbow_silhouette(X_pca)
        eps_range = np.arange(args.eps_range_start, args.eps_range_end, args.eps_range_step)
        perform_clustering(X_pca, k, df, args.meta)
    except ValueError as ve:
        print(f"[ERROR] {ve}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
