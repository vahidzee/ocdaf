
# Create your custom Benchmark

*TODO*: add some explanation on how to create a new dataset and reference it in the YAML file.

# Use our synthetic benchmark code

*TODO*: add a simple guide for this 

# Benchmarking datasets

In this section, we explain the meta-configurations relating to our benchmarking experiments reported in the paper:


## Sachs dataset

We have created a sweep configuration [here](./meta_configurations/sachs.yaml) that runs the Sachs dataset on different seeds using the `seed_everything` option that Lightning provides. This will automatically create a sweep that runs the Sachs dataset on different hyper-parameter configurations, and for each configuration, it will run it for five different seeds.

Finally, the run will produce a set of model results as `json` files in the `experiments/saves/sachs` (TODO: fix) directory. These `json` files will contain full detail of the final ordering that the model has converged to and it can then later on be used for pruning.

## Syntren dataset
Similar to Sachs, the sweep configuration for this run is available [here](./meta_configurations/syntren.yaml). This is a simple sweep that will run all of the Syntren datas (with identifiers ranging from 0 to 1) and produce the same set of result `json` files in `experiments/saves/syntren` (TODO: fix).

## Synthetic datasets

We provide several sweep configurations for synthetic datasets, each covering a specific set of conditions and scenarios. The results are conveniently summarized using the Weights and Biases UI.

### Small parametric datasets

The configuration for these experiments can be found [here](./meta_configurations/synthetic-param-gaussian-small.yaml). It covers graphs with 3, 4, 5, and 6 covariates generated by different algorithms (tournaments, paths, and Erdos-Renyi graphs). The functional forms included are sinusoidal, polynomial, and linear, all accompanied with Gaussian noise. For a comparative study between affine and additive, both options are also included. Each configuration is run five times with different seeds. 

We test each dataset using three algorithms: Gumbel top-k, Gumbel Sinkhorn, and Soft. In total, this sweep contains 1480 different configurations.

### Small non-parametric datasets

You can find the sweep configuration for these datasets [here](./meta_configurations/synthetic-non-param.yaml). Similar to the parametric configuration, it covers graphs with 3, 4, 5, and 6 covariates. However, these datasets are generated using Gaussian processes to sample the scale and shift functions. Both Affine and Additive options are included for comparison, and each set of configuration is also seeded 5 times, totalling to 240 different configurations.

### Small Linear Laplace Datasets

The configuration for the linear Laplace runs can be found [here](./meta_configurations/synthetic-linear-laplace.yaml). This experiment demonstrates that our model can handle broader classes of Latent Structural Nonlinear Models (LSNMs), providing insights into possible updates of our theoretical conditions. For these configurations, we use small graphs with different generation schemes, but we employ a linear function for the scale and shift and choose a standard Laplace noise. The number of configurations generated by this sweep on different seeds totals to 480 runs.