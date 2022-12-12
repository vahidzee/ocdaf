# Ordered Causal Discovery (OCD)

This library is based on [lightning-toolbox](https://github.com/vahidzee/lightning-toolbox) and [dycode](https://github.com/vahidzee/dycode) which we developed to enable dynamic training procedures, and model designs in our experiments.

## Setup
In a python (>=3.9) environment, run the following commands:
```bash
git clone https://github.com/vahidzee/ocd.git # clone the repository
cd ocd
pip install -r requirements.txt # install dependencies
```

## Reproducing the experiments

Notebooks for reproducing the experiments in the paper are available in the `notebooks` folder:
- [Visualizing Birkhoff Polytope](notebooks/birkhoff.ipynb)
- [Sanity check for log-likelihoods](notebooks/sanity_check.ipynb)
- [Pruning of causal orders](notebooks/pruning.ipynb)
- [Ordered Causal Discovery with MADE](notebooks/ocd.ipynb)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Citation

If you use OCD in your research, please cite this repository as described in [CITATION.cff](CITATION.cff).

