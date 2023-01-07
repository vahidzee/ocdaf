"""
This file contains a callback that visualizes the explorability of the model.

The explorability of the model is a concept based on the distribution of permutations that 
are generated by the model at each phase.

We expect the model starting off exploring a lot of permutations and then ends up concentrating
on a key set of permutations that resemble a correct ordering. To do that, we use a visualization
of the permutation matrices using the doubly stochastic Birkhoff polytope.

In the beginning we will train a PCA on the Birkhoff polytope of some size, and then, after 
each phase change we will get all the logged permutations of that phase, feed it to the pre-trained
PCA and do a scatter plot to show the distribution of the permutations in that phase.
"""

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ocd.models.permutation.utils import sinkhorn
import torch
from itertools import permutations
import typing as th
from ocd.data.scm import SCM


MARKERS = ["^", "o", "x"]


def get_core_points(
    permutation_size: int,
    num_points: int,
    birkhoff_edges: bool = False,
    birkhoff_vertices: bool = True,
):
    """
    Given a permutation_size, this function returns the core points of the Birkhoff polytope.

    The core points can be the following:
    1. The permutation matrices of size permutation_size, this will correspond to the vertices
        of the Birkhoff polytope.
    2. The mid-way points between the permutation matrices, this will correspond to the edges
        of the Birkhoff polytope.

    After listing all the core points according to the priority provided above, we will then sample
    num_points from the list of core points. This sampling is done iteratively so that each core point
    will have the most distance from the previous set of points.

    Args:
        permutation_size: the size of the permutation matrices
        birkhoff_edges: whether to include the mid-way points between the permutation matrices
        birkhoff_vertices: whether to include the permutation matrices

    Returns:
        core_points: a tensor of shape (num_core_points, permutation_size, permutation_size)
    """

    # create all the permutation matrices of size permutation_size
    # and append them to the core_points
    core_points = None

    # add birkhoff_vertices to the core_points
    if birkhoff_vertices:
        for perm in permutations(range(permutation_size)):
            # create a permutation matrix out of perm
            perm_mat = np.zeros((permutation_size, permutation_size))
            perm_mat[np.arange(permutation_size), perm] = 1
            # extend perm_mat in the first dimension
            perm_mat = np.expand_dims(perm_mat, axis=0)
            # concatenate perm_mat to the birkoff_vertices
            core_points = perm_mat if core_points is None else np.concatenate([core_points, perm_mat], axis=0)

    # add birkhoff_edges to the core_points
    if birkhoff_edges:
        edge_points = []
        for perm1 in permutations(range(permutation_size)):
            for perm2 in permutations(range(permutation_size)):
                # create a permutation matrix out of perm
                perm_mat1 = np.zeros((permutation_size, permutation_size))
                perm_mat1[np.arange(permutation_size), perm1] = 1
                # create a permutation matrix out of perm
                perm_mat2 = np.zeros((permutation_size, permutation_size))
                perm_mat2[np.arange(permutation_size), perm2] = 1
                # create the edge point
                edge_point = (perm_mat1 + perm_mat2) / 2
                # append the edge point to the edge_points
                edge_points.append(edge_point)
        # concatenate all the edge points to the core_points
        t = np.stack(edge_points, axis=0)
        core_points = t if core_points is None else np.concatenate([core_points, t], axis=0)

    # Now iteratively sample num_points from the core_points
    sampled_core_points = None
    num_points = min(num_points, core_points.shape[0])
    for i in range(num_points):
        # if this is the first point, just sample the first point
        if i == 0:
            sampled_core_points = core_points[0]
            # expand the sampled_core_points in the first dimension
            sampled_core_points = np.expand_dims(sampled_core_points, axis=0)
        else:
            set_a = sampled_core_points.reshape(sampled_core_points.shape[0], -1)
            set_b = core_points.reshape(core_points.shape[0], -1)
            # get an i x core_points.shape[0] matrix of distances
            # between the sampled_core_points and the core_points
            dist = np.linalg.norm(
                np.expand_dims(set_a, axis=1) - np.expand_dims(set_b, axis=0),
                axis=2,
            )
            # for each core_point corresponding to the columns in dist,
            # get the minimum distance from the sampled_core_points
            dist = np.min(dist, axis=0)

            # get the column with the maximum distance
            idx = np.argmax(dist)
            # add the point with the maximum distance to the sampled_core_points
            sampled_core_points = np.concatenate(
                [sampled_core_points, np.expand_dims(core_points[idx], axis=0)], axis=0
            )

    # return the sampled_core_points
    return sampled_core_points


def cluster_particles(all_points: np.array, core_points: np.array) -> np.array:
    """
    This function clusters all the particles according to how close they are to
    core_points

    Args:
        all_points: [n_samples, permutation_size, permutation_size]
        core_points: [n_core_points, permutation_size, permutation_size]
    Returns:
        clusters: a one dimensional np.array of size n_samples that assigns each point to
                    a cluster
    """
    # cluster doubly_stochastic_matrices according to what index of
    # core_points they are closest to
    clusters = np.zeros(all_points.shape[0])
    for i, mat in enumerate(all_points):
        # get the index of the closest vertex
        closest_vertex = np.argmin(np.linalg.norm(core_points - mat, axis=(1, 2)))
        clusters[i] = closest_vertex
    return clusters


def get_birkhoff_samples(permutation_size: int, n_sample: int = 100) -> np.array:
    """
    Args:
        permutation_size: the size of the permutation matrices
        n_sample: the number of samples to draw from the polytope
    Returns:
        polytope: a tensor of shape (n_sample, permutation_size, permutation_size) as a numpy array
    """
    # sample n_sample x permutation_size x permutation_size gumbel noises
    gumbel_noise = np.random.gumbel(size=(n_sample, permutation_size, permutation_size))
    # turn the gumbel noise into a torch tensor
    gumbel_noise = torch.from_numpy(gumbel_noise).float()
    polytope = (
        sinkhorn(torch.cat([gumbel_noise, gumbel_noise / 0.1, gumbel_noise / 0.05], dim=0), 100).detach().numpy()
    )
    return polytope


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


class BirkhoffCallback(Callback):
    def __init__(
        self,
        permutation_size: int,
        seed: th.Optional[int] = None,
        # logging frequencies
        log_on_phase_change: bool = True,
        log_every_n_epochs: int = 0,
        clear_logs_each_epoch: bool = True,
        # PCA setting
        fit_every_time: bool = False,
        # loss values printed
        write_cost_values: bool = False,
        loss_cluster_count: int = 100,
        core_points_has_birkhoff_vertices: bool = True,
        core_points_has_birkhoff_edges: bool = False,
        # Including correct orderings
        scm: SCM = None,
        # Include permutation names
        add_permutation_to_name: bool = False,
    ) -> None:
        """
        This is a lightning callback that visualizes how the model explores and behaves.
        In summary, it visualizes all the latent permutations that are sampled at each of the maximization
        and expectation steps. To do so, we use PCA on the Birkhoff polytope to project the permutations
        to 2D. We then visualize the 2D projections of the permutations.

        Using the following argument you can control the logging process:
            log_on_phase_change: whether to log the explorability when the phase of the training module changes
            log_every_n_epochs: whether to log the explorability every n epochs

        Args:
            permutation_size: the size of the permutation
            seed: the seed to use for the numpy random number generator
            log_on_phase_change: whether to log the explorability when the phase changes
            log_every_n_epochs: whether to log the explorability every n epochs
            fit_every_time: whether to fit the PCA every time or not
            write_loss_on_vertices: whether to write the loss on vertices or not
            loss_on_vertices_count: the number of vertices to write the loss on
            clear_logs_each_epoch: whether to clear the logs each epoch or not
            scm: the SCM object to use for finding the points associated with each ordering
            add_permutation_to_name: If this is set to true, then the average of each cluster
                                     is written in the legend as an approximate permutation
        """
        self.permutation_size = permutation_size
        self.last_saved_phase = None

        self.log_on_phase_change = log_on_phase_change
        self.log_every_n_epochs = log_every_n_epochs
        self.log_every_n_epochs_counter = 0
        self.clear_logs_each_epoch = clear_logs_each_epoch
        self.fit_every_time = fit_every_time
        self.pca = PCA(n_components=2)

        self.add_permutation_to_name = add_permutation_to_name

        self.seed = seed
        # set the seed of numpy
        np.random.seed(self.seed)

        # If we do not want to fit each and every time, then we need to sample
        # a Birkhoff polytope as a palette to start with and fit a core
        # PCA on it to visualize everything in two dimensions

        self.polytope = None
        self.transformed_polytope = None

        if not self.fit_every_time:
            self.polytope = get_birkhoff_samples(permutation_size)
            # train a PCA on all the elements of the polytope
            self.pca.fit(self.polytope.reshape(-1, permutation_size * permutation_size))
            self.transformed_polytope = self.pca.transform(
                self.polytope.reshape(-1, permutation_size * permutation_size)
            )

        # If we have to log the losses as well, we should get a set of core points
        # and save them
        self.write_cost_values = write_cost_values
        if self.write_cost_values:
            self.cluster_count = loss_cluster_count
            self.core_points = get_core_points(
                permutation_size,
                self.cluster_count,
                birkhoff_vertices=core_points_has_birkhoff_vertices,
                birkhoff_edges=core_points_has_birkhoff_edges,
            )

        # For each of the vertex points which are the permutation
        # set their delimiters according to the number of backward edges
        # they have. If something has a low number of backward edges, then
        # it will have a larger delimiter. This is used to determine the correct
        # orderings
        self.birkhoff_vertices = get_core_points(
            permutation_size, permutation_size**permutation_size, birkhoff_edges=False, birkhoff_vertices=True
        )

        self.birkhoff_vertex_scores = []
        for perm in self.birkhoff_vertices:
            if scm is None:
                self.birkhoff_vertex_scores.append(1)
            else:
                ordering = perm.argmax(-1).tolist()
                self.birkhoff_vertex_scores.append(scm.count_backward(ordering))
        self.birkhoff_vertex_scores = np.array(self.birkhoff_vertex_scores)

    def _print_unique_permutations(self, logged_permutations):
        real_logged_permutations = logged_permutations.argmax(axis=-1)
        # get the unique rows and the number of times they appear
        unique_rows, counts = np.unique(real_logged_permutations, axis=0, return_counts=True)
        print("Permutations that were seen:")
        for row, count in zip(unique_rows, counts):
            print(row, " : ", count, " times")

    def check_should_log(self, pl_module: pl.LightningModule) -> bool:
        """
        This function checks whether the callback should take action or not.
        Although the callback is called once every epoch, it does not necessarily
        log every epoch. This function checks whether the callback should log
        according to the logging controls which are for example, log_on_phase_change
        and log_every_n_epochs.
        """
        # If the logging is not at the end of the phase change then
        # check frequency and return accordingly
        if not self.log_on_phase_change:
            t = self.log_every_n_epochs_counter
            self.log_every_n_epochs_counter = (t + 1) % self.log_every_n_epochs
            return t == self.log_every_n_epochs - 1

        # If the last phase is the same as the current phase, do nothing
        # This indicates no phase change edge
        if pl_module.get_phase() == self.last_saved_phase:
            return False
        # Do nothing if the last phase was None
        # (This means this is the first time that it is being called)
        if self.last_saved_phase is None:
            self.last_saved_phase = pl_module.get_phase()
            return False
        # A phase change has happened and now is the time to visualize the explorability
        self.last_saved_phase = pl_module.get_phase()
        return True

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        ret = super().on_train_epoch_end(trainer, pl_module)

        # get all the logs
        all_logs = pl_module.get_logged_input_outputs()

        # if we are to clear logs at each epoch clear them
        if self.clear_logs_each_epoch:
            pl_module.clear_logged_input_outputs()

        # check should log or not and if so just return the ret value
        # from parent and leave as is
        if not self.check_should_log(pl_module):
            return ret

        # get the logged permutations
        logged_permutations = all_logs["latent_permutation"].numpy()

        # If we are to train the PCA every time, then we should fit it with the logged permutations here
        if self.fit_every_time:
            self.pca.fit(logged_permutations.reshape(-1, self.permutation_size * self.permutation_size))

        #######################
        # All the points to log:
        #######################

        permutation_without_noise = pl_module.model.permutation_model.soft_permutation()
        permutation_without_noise = permutation_without_noise.detach().numpy()

        clusters = None
        cost_values = None
        if self.write_cost_values:
            clusters = cluster_particles(logged_permutations, self.core_points)
            cost_values = -all_logs["log_prob"].detach().numpy()

        img = visualize_exploration(
            visualization_model=self.pca,
            backbone=self.transformed_polytope,
            backbone_is_transformed=True,
            sampled_permutations=logged_permutations,
            clusters=clusters,
            cost_values=cost_values,
            permutation_without_noise=permutation_without_noise,
            birkhoff_vertices=self.birkhoff_vertices,
            birkhoff_vertices_cost=self.birkhoff_vertex_scores,
            add_permutation_to_name=self.add_permutation_to_name,
            colorbar_label="count backwards",
            image_size=(15, 10),
            title="Birkhoff Polytope of Permutations",
            ylabel=f"phase {pl_module.get_phase()}",
            xlabel=f"epoch: {pl_module.current_epoch}",
        )
        # get the root tensorboard logger
        logger = pl_module.logger.experiment
        logger.add_image(f"explorability/birkhoff", img, pl_module.current_epoch, dataformats="HWC")

        return ret
