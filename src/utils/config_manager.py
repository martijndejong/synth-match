import yaml

from abc import ABC
from collections.abc import Mapping


class Config(Mapping, ABC):
    """
    Config class that behaves like a dictionary and includes methods to load and save YAML configs.
    """
    def __init__(self, config_dict=None):
        """
        Initialize the Config object with an optional dictionary.
        """
        self._config = config_dict or {}

    def load(self, config_path: str):
        """
        Load the configuration from a YAML file.
        """
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def save(self, save_path: str):
        """
        Save the current configuration to a YAML file.
        """
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f)

    def __getitem__(self, key):
        """
        Allow direct dictionary-like access to the configuration.
        """
        return self._config[key]

    def __setitem__(self, key, value):
        """
        Allow direct dictionary-like setting of the configuration.
        """
        self._config[key] = value

    def __contains__(self, key):
        """
        Allow use of the `in` keyword with the configuration.
        """
        return key in self._config

    def __iter__(self):
        """
        Make the Config object iterable (e.g., for wandb or other libraries expecting a dict).
        """
        return iter(self._config)

    def __repr__(self):
        """
        Represent the Config object with its dictionary contents.
        """
        return repr(self._config)

    def items(self):
        """
        Return items from the underlying config dictionary.
        """
        return self._config.items()

    def as_dict(self):
        """
        Return the underlying configuration as a dictionary.
        """
        return self._config
