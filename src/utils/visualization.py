import matplotlib.colors as mcolors
import umap.umap_ as umap


def plot_pca_tsne_pids(title, naming, model, test_loader, num_video=15, num_elements=600, show_vid=True, load_model=True):
    """
    Plots PCA and t-SNE visualizations for the given model and test data.

    This function takes a trained model and a test data loader, performs dimensionality reduction
    using PCA and t-SNE, and plots the results. The plots are saved as a PNG file with the specified naming.

    Args:
        title (str): The title to be printed before plotting.
        naming (str): The base name for the saved plot file.
        model (torch.nn.Module): The trained model used for feature extraction.
        test_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
    """
    print()
    print(title)
    print()
    model=model.to(device)
    model.eval()
    X = []
    y = []
    pids = []
    with torch.no_grad():
        # pred = []
        for el in test_loader:
            if load_model:
                X += model(el['data'].to(device)).cpu().tolist()
            else:
                X += el['data'].reshape(-1, el['data'].shape[-1]*el['data'].shape[-2]*el['data'].shape[-3]).tolist()
            
            y += el['label'].tolist()
            pids += el['patient'].tolist()
            # pred += ft_model(el['data'].to(device)).cpu().tolist()
            # if len(pids) > num_elements:
            #     p, c = np.unique(pids, return_counts=True)
            #     if min(c) >= num_elements // num_video:
            #         break

    if len(X) > num_elements:
        X, _, y, _, pids, _ = train_test_split(X, y, pids,
                        train_size=num_elements, random_state=21, shuffle=True)

    X = np.array(X)
    y = np.array(y)
    if show_vid:
        pids = np.array(pids).astype(int) % num_video
    else:
        pids = np.array(pids).astype(int) // num_video

    # print(X.shape, pids.shape)
    # print(max(pids))
    # print(num_video)
    # print(pids)

    if num_video == 15:
        # cmaping = np.concatenate([["forestgreen",  "firebrick", "dodgerblue",
        #         "skyblue", "darkred",  "limegreen",
        #         "aqua", "indianred",  "darkgreen",
        #         "mediumseagreen",  "salmon","teal",
        #         "red", "lightgreen", "lightblue" ] for _ in range(len(y) // num_video)])
        cmaping = np.array(["forestgreen",  "firebrick", "dodgerblue",
                "skyblue", "darkred",  "limegreen",
                "aqua", "indianred",  "darkgreen",
                "mediumseagreen",  "salmon","teal",
                "red", "lightgreen", "lightblue" ] )
        cmaping = cmaping[pids]
    else:
        cmaping = np.array([i for i in list(mcolors.CSS4_COLORS.keys()) if ("white" not in i) and ("black" not in i)][:num_video])
        cmaping = cmaping[pids]
    # print(cmaping.shape)

    random_state = 0

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))

    # Reduce dimension to 2 with PCA
    pca = PCA(n_components=2, random_state=random_state)

    # Fit the method's model
    # Embed the data set in 2 dimensions using the fitted model
    X_embedded = pca.fit_transform(X)

    # Plot the projected points and show the evaluation score
    ax2.scatter(X_embedded[:, 0], X_embedded[:, 1], c=cmaping, s=30)
    ax2.set_title("PCA "+naming)

    ax1.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap="brg")
    ax1.set_title("PCA "+naming)

    X_embedded = TSNE(n_components=2, learning_rate='auto', method='exact', metric='cosine',
                    init='pca', perplexity=10).fit_transform(X)

    ax4.scatter(X_embedded.T[0], X_embedded.T[1], c = cmaping)
    ax4.set_title("t-SNE "+naming)

    ax3.scatter(X_embedded.T[0], X_embedded.T[1], c = y, cmap='brg')
    ax3.set_title("t-SNE "+naming)

    fig.savefig(naming+".png")
    plt.show()
