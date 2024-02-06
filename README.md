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
    Or, you can opt for a conda environment:
    ```
    conda env create -f env.yml # This will create a conda environment named oslow which you can activate
    ```

### Experiment Configuration

All experiments are defined and reproducible using a hierarchical `yaml` configuration encompassing:

1. The dataset for causal discovery.
2. The flow-based model (`torch.nn.Module` implementation).
3. The trainer (optimization, scheduling, epoch count details).

Though configurations are adjustable down to minute details, we provide:

- A lightweight [sample configuration](./configurations/examples/simple.yaml) based on our paper's best results.
- A detailed [full configuration](./configurations/examples/full.yaml) for deeper insights into the training pipeline.

This consolidated `yaml` approach enhances transparency for deep learning model development and assists future developers in navigating the nuances of our setup.

## Sweeps

Run sweeps using the following command:
```bash
wandb sweep sweep/<sweep-config>.yaml --project <project-name> (optional) --entity <workspace-name>
```

## Interventions
To reproduce the results of our interventional experiments, look at [this notebook](./experiments/intervention/results.ipynb) for further instructions.
The resulting checkpoints of the trained models are also available [here](./experiments/intervention/checkpoints/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
