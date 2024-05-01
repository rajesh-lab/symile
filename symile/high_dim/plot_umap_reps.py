import numpy as np
import matplotlib.pyplot as plt

def plot_embeddings(file_a, file_i, file_t, save_dir):
    e_a = np.load(f"{save_dir}/{file_a}.npy")
    e_i = np.load(f"{save_dir}/{file_i}.npy")
    e_t = np.load(f"{save_dir}/{file_t}.npy")

    plt.figure(figsize=(10, 8))
    plt.scatter(e_a[:, 0], e_a[:, 1], color='red', alpha=0.5, label='r_a')
    plt.scatter(e_i[:, 0], e_i[:, 1], color='blue', alpha=0.5, label='r_i')
    plt.scatter(e_t[:, 0], e_t[:, 1], color='green', alpha=0.5, label='r_t')

    # Optionally connect triples with lines
    for j in range(e_a.shape[0]):  # assuming all are the same length
        plt.plot([e_a[j, 0], e_i[j, 0], e_t[j, 0]], [e_a[j, 1], e_i[j, 1], e_t[j, 1]], 'k-', alpha=0.5)

    plt.title('UMAP Visualization of Triple Embeddings')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    save_dir = "/gpfs/scratch/as16583/symile/symile/high_dim/results/umap"
    plot_embeddings('r_a', 'r_i', 'r_t', save_dir)