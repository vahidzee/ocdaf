# Instructions to Run the Trainers

For running our experiments, we use an argument parser inspired by [LightningCLI](https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html) which uses a [jsonargparse](https://jsonargparse.readthedocs.io/en/stable/) formatting to parse arguments either from a yaml config file or explicitely from the command line.

## Simple Trainer

We follow the simple LightningCLI conventions for the simple trainer. The simple trainer is run using the `trainer.py` file in the root directory. To run this, you have to explicitly define the configurations for the model, trainer, and the datamodule to work. Some default configurations can be found in the `experiments/simple` directory.
For example, `data-simple.yaml` contains configurations for a 3-covariate dataset with 1000 samples. `model-simple.yaml` contains configurations for a simple ODCAF model and `trainer-simple.yaml` contains configurations for the trainer which contains the Birkhoff Polytope callback and checkpointing. Run the following in the root directory to get the results for the simple trainer:

```bash
python trainer.py fit --data <path-to-data> --model experiments/simple/<model-config-file>.yaml --trainer experiments/simple/<trainer-config-file>.yaml
```

or 
```bash
python trainer.py fit --config <config-file>.yaml
```

For the templates for example, you can re-write as follows:

```bash
python trainer.py --data experiments/simple/data-simple.yaml --model experiments/simple/model-simple.yaml --trainer experiments/simple/trainer-simple.yaml
```

## Smart Trainer

The smart trainer is a code that controls the meta-data of the simple trainer. If you work with the simple trainer a little bit, you will notice that some fields are redundant accross data and model. Some examples of logics that apply accorss different configurations include:
1. If you have a data generating process that uses Gaussian noise as the base noise distribution, you would want to set your OCDaf model to use Gaussian noise as base distribution as well. Therefore, there is a dependency between the `data-config.yaml` file and `model-config.yaml` file. The smart trainer checks the data configuration and overwrites the model configuration accordingly.
2. For a smaller number of covariates, you might want to create a Birkhoff callback that visualizes the training process in its entirity. You would notice that to do so, you would require to pass in the actual ground truth causal graph that is obtained from the data configuration to the causal_graph argument of the Birkhoff callback. Therefore, there is a dependency between the `data-config.yaml` file and `trainer-config.yaml` file. The smart trainer checks the data configuration and overwrites the trainer configuration accordingly.
3. With the number of covariates increasing, the architecture should be increasingly complex. Therefore, the smart_trainer first of all auto-filles the `in_features` argument of the model with the value obtained from the data configuration, and then, it multiplies each of the hidden dimensions with this value. This way, the model architecture is automatically scaled with the number of covariates.

All the logs of the smart trainer are saved in the `experiments/smart-trainer-log` directory. The smart trainer creates a subdirectory under this directory and creates a `causal_discovery-config.yaml` which is the final configuration that the **Simple** trainer is being run on. Note that to run the smart trainer, you can use the same scheme as the simple trainer but drop the `fit` argument (as it is automatically set to fit) and also add a `--discovery` option to it.

```bash
python smart_trainer.py --data <data-config-file>.yaml --model <model-config-file>.yaml --trainer <trainer-config-file>.yaml --discovery
```

You can find a set of template configurations in the `experiments/smart` directory to play around with.

### Smart Trainer with Causal Inference

The smart trainer can also be used to do a two-step training phase. If you add a `--inference` option to the runner above, after running the discovery phase, it will read the correct permutation from the `json` logs that the model provides and using that correct ordering fits a new model. This way, you can use the smart trainer to do causal inference as well by facilitating the entire parameters of OCDaf and running it on the basic fixed permutation setting. The model obtained from this phase can be used for inference purposes such as modeling interventions or generating data.

```bash
python smart_trainer.py --data <data-config-file>.yaml --model <model-config-file>.yaml --trainer <trainer-config-file>.yaml --discovery --inference
```

## Sweep

Our implementation of sweep uses the classic sweep from [Weights and Biases](https://docs.wandb.ai/guides/sweeps) but extends it to handle nested hyper-parameter tuning as well.

Sweep is the king of all experiments! It can be used for all kinds of parallel experimentation and hyper-parameter tuning. To run a sweep, you should run the `sweep.py` file which follows a similar scheme as the trainers we have mentioned, with the added `--sweep` option. The `--sweep` option can contain a set of configurations for a `SweepObject` described as below:

```python
class Sweep:
    project: str # The sweep project name
    agent_run_args: # This contains all the additional arguments that will be passed to the wandb.agent function
        count: int 
    sweep_configuration: 
        method: ['grid', 'random', 'bayes']
        metric: str # something to monitor and optimize
        parameters: SweepNestedParameters # An entire nested dictionary of SweepParameter
    run_name: str # Some default name or None
    checkpoint_interval: int # The number of seconds for checkpointing to occure
    default_root_dir: str # The path that the sweep works in, by default it is set to "experiments/sweep" which contains all the logs and checkpoints
    sweep_id: Optional[str] # Either a sweep_id for when you want to dispatch an agent, or None when you want to simply create a sweep object
```

While the conventional sweep has a flat parameters argument, the sweep that we have implemented can also handle nested structures. We will get back to this later. Note that when you run the sweep without a `sweep_id` it will simply create a sweep project to work with. 
```bash
python sweep.py --data <data-config-file>.yaml --model <model-config-file>.yaml --trainer <trainer-config-file>.yaml --sweep <sweep-config-file>.yaml
```

When you have done that and have obtained a sweep id from weights and biases, you can run the sweep with the following configuration:
```bash
python sweep.py --data <data-config-file>.yaml --model <model-config-file>.yaml --trainer <trainer-config-file>.yaml --sweep <sweep-config-file>.yaml --sweep.sweep_id=<sweep-id>
```
Moreover, since the sweep option also follows the `jsonargparse` standard, you can reset each of the parameters of the sweep object by passing them as arguments. For example, you can set a default run_name by adding `--sweep.run_name=<run-name>` to the command above.

All in all, the sweep object creates a configuration from the nested parameter setup that you have given to it and **overwrites** the configurations that have been given by the `--data`, `--model`, and `--trainer`. After overwriting, it will simply work as if it has passed all the new meta-data to the **Simple Trainer**. To use the smart trainer, you can set `--sweep.use_smart_trainer=True`.

```bash
python sweep.py --data <data-config-file>.yaml --model <model-config-file>.yaml --trainer <trainer-config-file>.yaml --sweep <sweep-config-file>.yaml --sweep.sweep_id=<sweep-id> --sweep.use_smart_trainer=True
```
You can also write this in the sweep configuration file itself.

### Nested Parameter Sweep

Up until now, we have described how to run the sweep. Here, we will describe what to setup in the `sweep.sweep_configuration.parameters` for the sweep to work. You can write this field as if you are writing a simple configuration for the smart_trainer or the simple trainer. The only difference is that you can set up different values or options for specific fields. For example, say you want to run a sweep that considers different activation functions negative slopes and different max_epochs. You can setup the sweep configuration file as below:

```yaml
sweep_configuration:
    parameters:
        model:
            init_args:
                activation_args:
                    negative_slope: 
                        sweep: True
                        values:
                        - 0.001
                        - 0.01
                        - 0.1
        trainer:
            max_epochs:
                sweep: True
                values:
                - 100
                - 1000
                - 10000
```

Our sweep will simply create different values for them and overwrites the nested structure of the configuration file. An example sweep configuration can be found in the `experiments/sweep` directory.

**Visualization Aliases for Keys.** When you run such a sweep, our code will automaticall assign names to the nested values that are being sweeped upon. However, you can also put aliases for them in the visualization. For example, if you want the `negative_slope` to be shown as `ns` in the visualization, you can add the following to the sweep configuration file:

```yaml
sweep_configuration:
    parameters:
        model:
            init_args:
                activation_args:
                    negative_slope: 
                        unique_name: ns
                        sweep: True
                        values:
                        - 0.001
                        - 0.01
                        - 0.1
        trainer:
            max_epochs:
                sweep: True
                values:
                - 100
                - 1000
                - 10000
```

Note that the `unique_name` field should contain unique names for each of the sweeped parameters. If you do not provide a unique name, the code will throw an exception.

**Visualization Aliases for Values.** Sometimes, you would want to provide aliases for the values as well (This only works for sweeps that have `values` field). For example, you want to visualize `max_epochs=100` as `small_epochs`, `max_epoch=1000` as `medium_epochs`, and `max_epochs=10000` as `large_epochs`. You can do this by adding the following to the sweep configuration file:

```yaml
sweep_configuration:
    parameters:
        model:
            init_args:
                activation_args:
                    negative_slope: 
                        unique_name: ns
                        sweep: True
                        values:
                        - 0.001
                        - 0.01
                        - 0.1
        trainer:
            max_epochs:
                sweep: True
                values:
                - 100
                - 1000
                - 10000
                values_display_name:
                - small_epochs
                - medium_epochs
                - large_epochs
```

This can be particularly useful when the sweep on the values is not on simple primitive values, but on entire dictionaries or large strings.