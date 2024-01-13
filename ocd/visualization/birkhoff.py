from typing import Optional, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import torch
from ocd.models.oslow import OSlow
from functools import lru_cache
from sklearn.decomposition import PCA
from ocd.training.trainer import PermutationLearningModule

    

@lru_cache(maxsize=128)
def get_all_permutation_matrices(num_nodes: int) -> torch.Tensor:
    # return a tensor of shape (num_nodes!, num_nodes, num_nodes) where the first dimension is all the permutation matrices
    
  
    return torch.tensor(
        np.array(
            [
                np.eye(num_nodes)[list(p)]
                for p in np.ndindex(*([num_nodes] * num_nodes))
                if np.all(np.sum(np.eye(num_nodes)[list(p)], axis=0) == 1)
                and np.all(np.sum(np.eye(num_nodes)[list(p)], axis=1) == 1)
            ]
        )
    )
    
def get_label_from_permutation_matrix(permutation_matrix: torch.Tensor) -> str:
    return "".join([str(int(i)) for i in permutation_matrix.argmax(dim=-1)])

def visualize_birkhoff_polytope(
    permutation_model: PermutationLearningModule,
    num_samples: int,
    data: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    flow_model: Optional[OSlow] = None,
    device: str = "cpu",
):
    permutation_model.to(device)
    sampled_permutations = permutation_model.sample(num_samples)
      
    # (1) Handle the backbone and the PCA transform of the Birkhoff polytope
    d = sampled_permutations.shape[-1]
    all_permutation_matrices = get_all_permutation_matrices(d)
    # concatenate all_permutation_matrices with backbone and train a PCA
    backbone = all_permutation_matrices
    pca = PCA(n_components=2, random_state=42)
    pca.fit(backbone.reshape(backbone.shape[0], -1))
    
    # (2) quantize the sampled permutations and use unique
    num_bins = 100
    sampled_permutations = torch.round(sampled_permutations * num_bins) / num_bins
    
    
    # (3) find the closest reference permutation matrix for each sampled permutation
    flattened_samples = sampled_permutations.reshape(sampled_permutations.shape[0], -1)
    flattened_references = all_permutation_matrices.reshape(all_permutation_matrices.shape[0], -1)
    # for every flattened samples find the closest reference
    closest_references = torch.argmin(torch.norm(flattened_samples.unsqueeze(1) - flattened_references.unsqueeze(0), dim=-1), dim=-1)
    img_data = None
    
    try:
        # plot the log_prob values of the closest references using the data if available
        fig, ax = plt.subplots()
        
        # plot backbone without showing it in the legend 
        ax.scatter(pca.transform(backbone.reshape(backbone.shape[0], -1))[:, 0], pca.transform(backbone.reshape(backbone.shape[0], -1))[:, 1], alpha=0)
        
        if isinstance(data, torch.Tensor):
            data = [data]
        
        for i, permutation in enumerate(all_permutation_matrices):
            lbl = get_label_from_permutation_matrix(permutation)
            
            average_log_prob = 0
            if (closest_references == i).any():
                close_samples = sampled_permutations[closest_references == i]
                alpha_value = max(0.3, 1.0 / (closest_references == i).sum().item())
                ax.scatter(pca.transform(close_samples.reshape(close_samples.shape[0], -1))[:, 0], pca.transform(close_samples.reshape(close_samples.shape[0], -1))[:, 1], label=lbl, alpha=alpha_value)
                
                if data is not None and flow_model is not None:
                    flow_model.to(device)
                    with torch.no_grad():
                        log_prob_sum = 0
                        num_data = 0
                        for data_batch in data:
                            data_batch = data_batch.to(device)
                            log_prob_sum += flow_model.log_prob(data_batch.float(), perm_mat=permutation.float()).sum()
                            num_data += data_batch.shape[0]
                        average_log_prob = log_prob_sum / num_data 
            x, y = pca.transform(permutation.reshape(1, -1))[0, 0], pca.transform(permutation.reshape(1, -1))[0, 1]
            # count the number of entries in the closest references that is equal to i
            # write a text on (x, y) with value of average_log_prob
            ax.text(x, y, f"{average_log_prob:.2f} : #{(closest_references == i).sum()}", fontsize=8)
        
        mean_samples = torch.mean(sampled_permutations, dim=0)
        ax.scatter(pca.transform(mean_samples.reshape(1, -1))[:, 0], pca.transform(mean_samples.reshape(1, -1))[:, 1], label="mean", marker="x", color="red")
        ax.legend()
        
        # draw everything to the figure for conversion
        fig.canvas.draw()
        # convert the figure to a numpy array
        img_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    finally:
        plt.close()
    
    return img_data
        
    