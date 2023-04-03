import typing as th
import numpy as np


def visualize_exploration(
    visualization_model,
    sampled_permutations: np.array,
    backbone: th.Optional[np.array] = None,
    backbone_is_transformed: bool = True,
    clusters: th.Optional[np.array] = None,
    cost_values: th.Optional[np.array] = None,
    permutation_without_noise: th.Optional[np.array] = None,
    add_permutation_to_name: bool = False,
    birkhoff_vertices: th.Optional[np.array] = None,
    birkhoff_vertices_cost: th.Optional[np.array] = None,
    colorbar_label: str = "permutation scores",
    image_size: th.Optional[th.Tuple[float, float]] = None,
    ylabel: str = "y",
    xlabel: str = "x",
    title: str = "title",
):
    """
    This function visualizes the exploration of the model
    Args:
        visualization_model: typically a dimension reduction model that can help visualize the Birkhoff polytope in
                            2D, e.g., PCA, t-SNE, UMAP
        sampled_permutations: the sampled permutations of the model in that step
        backbone: A set of points from the Birkhoff polytope which can be used as backbone
        backbone_is_transformed: whether the backbone is transformed or not, if so, then we do not need to do dimension
                                reduction on the backbone
        clusters: the clusters of the sampled permutations, this is used to see what is the loss for example on each
                cluster
        cost_values: Each permutation has a cost assigned to it and this is used to visualize the cost of each cluster
        permutation_without_noise: the permutation without noise to check the state of the model parameters
        birkhoff_vertices: the birkhoff vertices themselves
        birkhoff_vertices_delimiters: the delimiters of the birkhoff vertices, these delimiters can be used to
                                    figure out which ordering should be better in terms of cost
        add_permutation_to_name: whether to add the permutation to the name of the image or not
        colorbar_label: the label of the colorbar  (default: "permutation scores")
        image_size: the size of the image
        ylabel: the label of the y axis
        xlabel: the label of the x axis
        title: the title of the image

    Returns:
        an image that can be saved
    """

    fig, ax = plt.subplots()

    # customize the image size if needed
    if image_size:
        fig.set_size_inches(image_size[0], image_size[1])

    try:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        # (1) plot the backbone
        if backbone is not None:
            backbone_t = backbone
            if not backbone_is_transformed:
                backbone_t = visualization_model.transform(backbone.reshape(backbone.shape[0], -1))
            ax.scatter(backbone_t[:, 0], backbone_t[:, 1], s=1, c="black", label="Backbone", alpha=0.1)

        # (2) plot the sampled permutations
        # Use clusters if it is not set to none
        sampled_permutations_t = visualization_model.transform(
            sampled_permutations.reshape(sampled_permutations.shape[0], -1)
        )
        if clusters is not None:
            for c in np.unique(clusters):
                # get the centroid of the cluster
                centroid = np.mean(sampled_permutations[clusters == c, :, :], axis=0)

                cluster_label = "cluster {} of samples".format(int(c + 1))
                if add_permutation_to_name:
                    cluster_label = f"{cluster_label} : {centroid.argmax(axis=-1)}"

                # set a marker according to cluster
                ax.scatter(
                    sampled_permutations_t[clusters == c, 0],
                    sampled_permutations_t[clusters == c, 1],
                    s=5,
                    label=cluster_label,
                )
                if cost_values is not None:
                    centroid_t = np.mean(sampled_permutations_t[clusters == c, :], axis=0)
                    # get the average cost of the cluster
                    cost = np.mean(cost_values[clusters == c])
                    text = f"{cost:.2f}/{len(cost_values[clusters == c])}"
                    # plot text on the centroid
                    ax.text(centroid_t[0], centroid_t[1], text, fontsize=8)
        else:
            ax.scatter(
                sampled_permutations_t[:, 0],
                sampled_permutations_t[:, 1],
                color="blue",
                s=5,
                label="Sampled doubly stochastics",
            )

        # (3) plot the birkhoff vertices using plt.scatter by
        # setting the delimiters according to birkhoff_vertices_delimiters
        if birkhoff_vertices is not None:
            birkhoff_vertices_t = visualization_model.transform(
                birkhoff_vertices.reshape(birkhoff_vertices.shape[0], -1)
            )
            plt.scatter(
                birkhoff_vertices_t[:, 0],
                birkhoff_vertices_t[:, 1],
                label=None,
                c=birkhoff_vertices_cost,
                cmap="brg",
                s=300,
                linewidth=0,
                alpha=0.5,
            )
            plt.colorbar(label=colorbar_label)

        # Plot the model parameters using the permutation without noise
        if permutation_without_noise is not None:
            permutation_without_noise_t = visualization_model.transform(
                permutation_without_noise.reshape(permutation_without_noise.shape[0], -1)
            )
            ax.scatter(
                permutation_without_noise_t[:, 0],
                permutation_without_noise_t[:, 1],
                s=300,
                c="red",
                marker="x",
                label="Gamma without noise",
            )
        ax.legend()

        # draw everything to the figure for conversion
        fig.canvas.draw()
        # convert the figure to a numpy array
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    finally:
        plt.close()

    return data
