import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# Load and preprocess data
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, sep='\t', header=0, index_col=0)
    X = df.values.astype(float)
    X_z = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    return X_z.T, df

# PCA and explained variance
def apply_pca(X, n_components=50):
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.show()
    return X_pca

# Elbow and silhouette analysis
def plot_elbow_silhouette(X, max_clusters=10):
    distortions, silhouettes = [], []
    Ks = range(2, max_clusters + 1)
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(X)
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
    plt.show()
    best_k = Ks[int(np.argmax(silhouettes))]
    print(f"Optimal k by silhouette: {best_k}")
    return best_k

# Tune DBSCAN eps parameter
def tune_dbscan(X):
    best_score, best_eps = -1, None
    for eps in range(1, 8):
        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
        mask = labels != -1
        if mask.sum() > 1:
            score = silhouette_score(X[mask], labels[mask])
            print(f"DBSCAN eps={eps} -> Silhouette: {score:.3f}")
            if score > best_score:
                best_score, best_eps = score, eps
        else:
            print(f"DBSCAN eps={eps} -> Not enough core samples")
    if best_eps is not None:
        print(f"Best DBSCAN eps: {best_eps} (Silhouette: {best_score:.3f})")
        return DBSCAN(eps=best_eps, min_samples=5).fit_predict(X)
    print("No valid eps found; marking all as noise")
    return np.full(X.shape[0], -1, dtype=int)

# Plot clustering results in 2D and 3D
def plot_clusters(X, labels, title):
    unique = np.unique(labels)
    plt.figure(figsize=(6, 5))
    for lab in unique:
        mask = labels == lab
        plt.scatter(X[mask, 0], X[mask, 1], label=str(lab), s=20, alpha=0.7)
    plt.title(f"{title} 2D")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()
    if X.shape[1] >= 3:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        for lab in unique:
            mask = labels == lab
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], s=20, alpha=0.7, label=str(lab))
        ax.set_title(f"{title} 3D")
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.legend(bbox_to_anchor=(1, 1))
        plt.show()

# Perform clustering and evaluation
def perform_clustering(X, k, df, meta_file):
    k_labels = KMeans(n_clusters=k, random_state=0, n_init='auto').fit_predict(X)
    print('K-Means:', silhouette_score(X, k_labels), calinski_harabasz_score(X, k_labels), davies_bouldin_score(X, k_labels))
    plot_clusters(X, k_labels, 'K-Means')
    g_labels = GaussianMixture(n_components=k, random_state=0).fit_predict(X)
    print('GMM:', silhouette_score(X, g_labels), calinski_harabasz_score(X, g_labels), davies_bouldin_score(X, g_labels))
    plot_clusters(X, g_labels, 'GMM')
    d_labels = tune_dbscan(X)
    mask = d_labels != -1
    if mask.sum() > 1:
        print('DBSCAN:', silhouette_score(X[mask], d_labels[mask]), calinski_harabasz_score(X[mask], d_labels[mask]), davies_bouldin_score(X[mask], d_labels[mask]))
    plot_clusters(X, d_labels, 'DBSCAN')
    meta = pd.read_csv(meta_file, sep='\t')
    true = meta.set_index('SAMPID').loc[df.columns]['SMTSD'].values
    print('ARI:', adjusted_rand_score(true, k_labels), 'NMI:', normalized_mutual_info_score(true, k_labels))
    emb = TSNE(n_components=2, random_state=0).fit_transform(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(emb[:, 0], emb[:, 1], c=k_labels, s=20)
    plt.title('t-SNE')
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
    plt.show()

# Main execution
file_exp = r'C:/Users/gogin/OneDrive/Desktop/applied machine learning/output_tissue_GTEX_exp1.txt'
meta_file = r'C:/Users/gogin/OneDrive/Desktop/applied machine learning/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt'
X, df = load_and_preprocess(file_exp)
X_pca = apply_pca(X)
k = plot_elbow_silhouette(X_pca)
perform_clustering(X_pca, k, df, meta_file)
