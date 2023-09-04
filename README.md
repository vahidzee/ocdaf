# Ordered Causal Discovery with Autoregressive Flows

![main_fig](https://github.com/vahidzee/ocdaf/assets/33608325/2352686b-965b-44d9-bd88-ee8b20ce7588)

<p align="center" markdown="1">
    <img src="https://img.shields.io/badge/Python-3.10-green.svg" alt="Python Version" height="18">
    <a href="https://arxiv.org/abs/2308.07480"><img src="https://img.shields.io/badge/arXiv-TODO-blue.svg" alt="arXiv" height="18"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#license">License</a>
</p>

Welcome to the official codebase for our research paper, [Ordered Causal Discovery with Autoregressive Flows](https://arxiv.org/abs/2308.07480). This work presents a cutting-edge method for causal discovery in multivariate Location Scale Noise Models (LSNMs) using autoregressive normalizing flows. Additionally, we've introduced a synthetic benchmark for causal discovery that caters to a diverse range of graph structures and functional forms within our Structural Causal Models (SCMs).

This repository is also integrated with Weights & Biases for comprehensive logging, hyperparameter tuning, and benchmarking. Through the [dysweep](https://github.com/HamidrezaKmK/dysweep) package, we ensure reproducibility and offer hierarchical configurations akin to [Hydra](https://hydra.cc/docs/intro/), but specifically designed for the Weights & Biases API.

## Getting Started

### Installation

1. Clone the repository:
    ```
    git clone https://github.com/vahidzee/ocdaf.git
    cd ocdaf
    ```

2. Within a Python (>=3.9) environment, install dependencies:
    ```
    pip install -r requirements.txt
    ```

    > Note: To maintain compatibility, consider using `frozen-requirements.txt` which lists the last confirmed working versions of the dependencies.

### Experiment Configuration

All experiments are defined and reproducible using a hierarchical `yaml` configuration encompassing:

1. The dataset for causal discovery.
2. The flow-based model (`torch.nn.Module` implementation).
3. The trainer (optimization, scheduling, epoch count details).

Though configurations are adjustable down to minute details, we provide:

- A lightweight [sample configuration](./configurations/examples/simple.yaml) based on our paper's best results.
- A detailed [full configuration](./configurations/examples/full.yaml) for deeper insights into the training pipeline.

This consolidated `yaml` approach enhances transparency for deep learning model development and assists future developers in navigating the nuances of our setup.

### Running Order Learning Experiments

1. Ensure you're logged into your Weights & Biases workspace as experiment logs will be displayed there. For setup, refer to the [quickstart](https://docs.wandb.ai/quickstart) guide.
   
2. Experiment configurations are located in the [configurations](./configurations/) directory. To reproduce a specific figure (e.g., Birkhoff polytope Figure 4) from our paper:

    ```
    python trainer.py fit --config configurations/examples/birkhoff-gumbel-sinkhorn.yaml --seed_everything=555
    ```

    The `--seed_everything` flag further bolsters reproducibility.

Every run generates an output `json` file following the format below: (TODO: add the format).

### Using Custom Datasets

To tailor this method to your datasets, we've crafted a step-by-step [guide](./DATA.md#create-your-custom-benchmark) to streamline the process.

### Pruning

Note that our causal discovery pipeline operates in a two-step manner, whereby it first determines the true ordering and then performs pruning. The pruning is essentially a post-processing step. For this, we provide a standalone script [prune.py](./prune.py) that handles the pruning process. By merging the `json` outputs from the order discovery phase with the pruning procedure, you get a complete causal structure discovery algorithm.

#### Example

Execute the pruning process with the following command:

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


## Benchmarking and Reproducibility

Our experimentation heavily relies on the [dysweep](https://github.com/HamidrezaKmK/dysweep) framework for extensive benchmarking. Our experiments are hierarchically organized in yaml files, and with dysweep, we can spawn multiple configuration sets for our experiments.

To initiate a sweep:

```bash
dysweep_create --config path/to/sweep-config.yaml
```
After generating a sweep object in your desired project, run configurations simultaneously on multiple machines:

```bash
dysweep_run_resume --package trainer --function dysweep_compatible_run --sweep_id <wandb-sweep-id> --count <no.-of-configurations-to-run-with-this-process>
```
To fully replicate our paper's experiments, refer to our six primary configuration sets within [the meta-configurations directory](./meta_configurations/) directory:

1. Sachs: These are runs related to the Sachs protein data, where we consider multiple runs of our model and report the average performance of our models. Check out the [sweep configuration](./meta_configurations/sachs.yaml) `yaml` file for more detail.
2. SynTReN: These are runs related to the synthetic gene expression data, and is our second dataset for highlighting the real-world capabilities of our model, we perform different runs on each of the data samples with different random seeds and report the average performance of our model. Check out the [sweep configuration](./meta_configurations/syntren.yaml) `yaml` file for more detail.
3. Synthetic Datasets: We consider different suites of synthetic dataset: (1) [small linear laplace SCMs](./meta_configurations/synthetic-linear-laplace.yaml), (2) [Small sinusoidal and polynomial SCMs](./meta_configurations/synthetic-param-gaussian-small.yaml) (parametric), (3) [Small Gaussian process inspired non-parametric SCMs](./meta_configurations/synthetic-non-param.yaml), and (4) [Large parametric and non-parametric datasets](./meta_configurations/synthetic-gaussian-big.yaml) for scalibility. 

For in-depth information on all six files, check [here](./DATA.md#benchmarking-datasets).

### Reproduce Pruning

We perform pruning to compare our results on the datasets Sachs and SynTReN. Use the command below to execute the pruning:

```bash
python prune.py --method=<cam/pc> --data_type=<syntren/sachs> --data_num=<data_id (Optional)> --order=<dash-separated-ordering> --saved_permutations_dir=<directory-to-saved-permutations>
```

To replicate our results:

1. Start the sweep for your target dataset. For instance, for the Syntren dataset, execute its corresponding sweep.
2. Post-sweep, the saved files will be stored in the `experiments/saves/syntren` directory. They'll be named following the `data-i` pattern, where `i` denotes the Syntren dataset's identifier.
3. With the saved files, execute CAM pruning for all Syntren datasets using the following command:

```bash
for i in {0..9}
do
    python prune.py --method=cam --data_type=syntren --data_num=$i --saved_permutations_dir=experiments/saves/syntren/data-$i
done
```
Consult the [results table](./configurations/real-world/prune_results.csv) to explore the various pruning techniques applied on the Sachs and SynTReN datasets.

## Interventions
To reproduce the results of our interventional experiments, look at [this notebook](./experiments/intervention/results.ipynb) for further instructions.
The resulting checkpoints of the trained models are also available [here](./experiments/intervention/checkpoints/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
