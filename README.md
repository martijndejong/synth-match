# synth-match
Attempt at using RL to generate synth patches from samples

## Poetry
Installing the specified dependencies:
```sh
poetry install
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