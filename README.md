# synth-match
Attempt at using reinforcement learning (RL) to generate synth patches from samples.

## Overview
SynthMatch is a project that uses reinforcement learning to train an agent to control an audio synthesizer to replicate a target sound by finding the right combination of synthesizer parameters. Unlike other methods that might rely on supervised learning, RL is used because the infinite space of continuous synthesizer parameter combinations makes direct prediction of these parameters impractical. The RL agent uses a control loop to iteratively adjust the synthesizer parameters based on feedback, effectively navigating this complex space to achieve the desired audio output.

Here is a screenshot of the agent in action, controlling a simple synthesizer to learn to reproduce a target sound:
![render_eaxmple_early_stage](https://github.com/martijndejong/synth-match/assets/12080489/68af3000-4ff3-4dd2-b7a4-0c05e3f019af)


## Agent
### Architecture
The system architecture divides the problem into two main components:
- **Observer Network**: This (convolutional) neural network observes and encodes the current and target audio into a meaningful feature space, facilitating the understanding of how close the synthesized sound is to the target.
- **Policy Network**: Based on the observer's output, this network suggests incremental changes to the synthesizer parameters. It is designed to optimize actions in a continuous space, directly influencing the synthesizer's controls.

### Modular Design
The separation into an observer and a policy network supports modularity, allowing each component to be updated independently, so that different observation methods and RL agent algorithms can easily be swapped in and out. This division also simplifies testing, maintenance, and scalability of the system.

## Environment
The RL agent interacts with a custom environment that houses a synthesizer object, this can be either a custom synthesizer or an existing VST plug-in. The environment processes the agent's actions (parameter adjustments), applies them to the synthesizer, and returns the resulting audio (spectrogram). It acts as the intermediary, translating the RL outputs into audible changes and vice versa. 


# Instructions

## Poetry
Creating venv with dependencies listed in pyproject.toml
```sh
poetry install
```

> **Note:** on Windows, when running the code after using Poetry to install the packages, you may get *ModuleNotFoundError: No module named 'tensorflow'* 
> 
>  To fix this, pip install TensorFlow into your venv using the following commands:
> ```sh
> poetry shell
> ```
> ```sh
> pip install tensorflow
> ```

Resolves and installs the latest compatible versions of dependencies.
```sh
poetry update
```

Adding packages that will be part of the final solution:
```sh
poetry add <package_name>
```

Adding packages that are only used for exploration/development:
```sh
poetry add --dev <package_name>
```


## Jupyter Notebooks
Exploration/manual testing can be done using Jupyter Notebooks. 
The .ipynb files should be saved in the /notebook folder.

Since not all editors support Jupyter Notebooks, we have added jupyterlab to the poetry --dev dependencies. 
To start up Jupyter Lab simply execute:
```sh
poetry run jupyter lab
```
