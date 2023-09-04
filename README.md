# Ordered Causal Discovery with Autoregressive Flows

![main_fig](https://github.com/vahidzee/ocdaf/assets/33608325/2352686b-965b-44d9-bd88-ee8b20ce7588)

<p align="center" markdown="1">
    <img src="https://img.shields.io/badge/Python-3.10-green.svg" alt="Python Version" height="18">
    <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv-TODO-blue.svg" alt="arXiv" height="18"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#license">License</a>
</p>

This repository provides the codebase for conducting experiments in our paper, built on top of the  [lightning-toolbox](https://github.com/vahidzee/lightning-toolbox) and [dypy](https://github.com/vahidzee/dypy). These tools facilitate dynamic training processes and model designs for our experiments. For large-scale benchmarking and hyperparameter tuning, we utilize the [dysweep](https://github.com/HamidrezaKmK/dysweep) package

## Installation
Begin by cloning the repository and navigating to the root directory. In a python (>=3.9) environment, run the following commands:
```bash
git clone https://github.com/vahidzee/ocdaf.git # clone the repository
cd ocdaf
pip install -r requirements.txt # install dependencies
```

To ensure compatibility, consider using the `frozen-requirements.txt`, which includes the versions of the dependencies last confirmed to work with our code.

## Experiments

The details for all the experiments mentioned in the paper can be found [`experiments`](https://github.com/vahidzee/ocdaf/tree/main/experiments/). Please read through the following for a big picture guide line on how to navigate the experimental details, and reproduce the results:
### Running Single Experiments

Single experiments can be conducted by defining a configuration file. For instance, the Birkhoff polytope figure in our paper can be reproduced with the following command:

```bash
python trainer.py fit --config experiments/configs/birkhoff-gumbel-sinkhorn.yaml --seed_everything=555
```

We have provided a sample configuration file with extensive documentation [here]((./experiments/examples/example-discovery.yaml)) to familiarize you with the components of our configurations. Furthermore, the `trainer.py` file is a standard [LightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html) runnable file that runs the causal discovery on a specific configuration defined. 

### Benchmarking

Our experiments leverage the framework described by [dysweep](https://github.com/HamidrezaKmK/dysweep) for large-scale benchmarking.
As mentioned, all of our experiments are given in a hierarchical format written as a `yaml` file and `dysweep` gives us the capability to generate different configurations to run our experiments. In particular, [the sweep directory]((./experiments/sweep)) contains a set of **meta-configurations** that generate a specific group of configurations which are then fed to our main trainer function.

To initiate a specific sweep, use the following command:

```bash
python sweep.py --config path/to/sweep-config.yaml
```

This generates a sweep object in the designated project with a unique ID. Subsequently, execute the following command across multiple machines to simultaneously run each configuration:

```bash
python sweep.py --entity=<Wandb-Entity> --project=<Wandb-Project> --sweep_id=<Wandb-Sweep-id> --count=<#-of-configurations-to-run>
```

Alternatively, simply add the `sweep_id` option to the initial command. Since we use `jsonargparse` this will simply rewrite the `sweep_id` on the previous configuration and seamlessly starts the sweep.

```bash
python sweep.py --config path/to/sweep-config.yaml --sweep_id=<Wandb-Sweep-id>
```

To completely reproduce our paper's experimental results, refer to the following sections:

#### Sachs dataset

We have created a sweep configuration [here](./experiments/sweep/sachs.yaml) that runs the Sachs dataset on different seeds using the `seed_everything` option that Lightning provides. This will automatically create a sweep that runs the Sachs dataset on different hyper-parameter configurations, and for each configuration, it will run it for five different seeds.

Finally, the run will produce a set of model results as `json` files in the `experiments/saves/sachs` directory. These `json` files will contain full detail of the final ordering that the model has converged to and it can then later on be used for pruning.

#### Syntren dataset
Similar to Sachs, the sweep configuration for this run is available [here](./experiments/sweep/syntren.yaml). This is a simple sweep that will run all of the Syntren datas (with identifiers ranging from 0 to 1) and produce the same set of result `json` files in `experiments/saves/syntren`.

#### Synthetic datasets

We provide several sweep configurations for synthetic datasets, each covering a specific set of conditions and scenarios. The results are conveniently summarized using the Weights and Biases UI.

#### Small parametric datasets

The configuration for these experiments can be found [here](./experiments/sweep/synthetic-param-gaussian-small.yaml). It covers graphs with 3, 4, 5, and 6 covariates generated by different algorithms (tournaments, paths, and Erdos-Renyi graphs). The functional forms included are sinusoidal, polynomial, and linear, all accompanied with Gaussian noise. For a comparative study between affine and additive, both options are also included. Each configuration is run five times with different seeds. 

We test each dataset using three algorithms: Gumbel top-k, Gumbel Sinkhorn, and Soft. In total, this sweep contains 1480 different configurations.

#### Small non-parametric datasets

You can find the sweep configuration for these datasets [here](./experiments/sweep/synthetic-non-param.yaml). Similar to the parametric configuration, it covers graphs with 3, 4, 5, and 6 covariates. However, these datasets are generated using Gaussian processes to sample the scale and shift functions. Both Affine and Additive options are included for comparison, and each set of configuration is also seeded 5 times, totalling to 240 different configurations.

#### Small Linear Laplace Datasets

The configuration for the linear Laplace runs can be found [here](./experiments/sweep/synthetic-linear-laplace.yaml). This experiment demonstrates that our model can handle broader classes of Latent Structural Nonlinear Models (LSNMs), providing insights into possible updates of our theoretical conditions. For these configurations, we use small graphs with different generation schemes, but we employ a linear function for the scale and shift and choose a standard Laplace noise. The number of configurations generated by this sweep on different seeds totals to 480 runs.

#### Large Datasets

For large synthetic datasets, the sweep configuration can be found [here](./experiments/sweep/synthetic-large.yaml). This set includes three different functional forms: sinusoidal, polynomial, and a Non-parametric scheme. The number of covariates is set to either 10 or 20, and each configuration is run on five different seeds. The final 30 synthetic configurations are passed on to the Gumbel-top-k method for evaluating model scalability.

You may refer to the [dysweep](https://github.com/HamidrezaKmK/dysweep) documentation to learn how to generate your own sweep configurations.

### Pruning

Our code also allows for pruning the final model ordering, which is facilitated by the `prune.py` file. Execute the pruning process with the following command:

```bash
python prune.py --method=<cam/pc> --data_type=<syntren/sachs> --data_num=<data_id (Optional)> --order=<dash-separated-ordering> --saved_permutations_dir=<directory-to-saved-permutations> 
```

In order to reproduce our results for the Sachs and Syntren datasets, you need to execute a series of steps after obtaining the `experiments/saves` directory:

1. un the sweep for the dataset you're interested in. For instance, if you're working on the Syntren dataset, execute the Syntren sweep.
2. After the sweep, a set of saved files will be available in the `experiments/saves/syntren` directory.
3. These files will follow the `data-i` format, where `i` represents the identifier of the Syntren dataset.
4. You can then use these saved files to run CAM pruning on all of the Syntren datasets. Run the command below, which iterates over all dataset IDs and performs pruning for each:

```bash
for i in {0..9}
do
    python prune.py --method=cam --data_type=syntren --data_num=$i --saved_permutations_dir=experiments/saves/syntren/data-$i
done
```
This process streamlines the replication of our results for the Sachs and Syntren datasets, using the CAM pruning method on all datasets generated by the sweep. Please check out the [results](./experiments/results/prune_results.csv) to check a table of the different pruning techniques on the Sachs and Syntren datasets. 

### Intervention
To reproduce the results of our interventional experiments, look at [this notebook](./experiments/intervention/results.ipynb) for further instructions.
The resulting checkpoints of the trained models are also available [here](./experiments/intervention/checkpoints/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
