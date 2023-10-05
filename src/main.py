import copy
import logging

from runtime import Runtime
from common import Params, load_configs, creat_unique_experiment_name
import fire

class EntryPoint(object):
    experiment = None
    configs = None

    def __init__(self, configs: str = "../configs/example.jsonnet"):

        _config = load_configs(configs)
        exp_name = creat_unique_experiment_name(_config)
        _config_copy = copy.deepcopy(_config)
        self.experiment = Runtime.from_params(Params(_config))
        self.experiment.setup(exp_name, _config_copy)

    def __getattr__(self, attr):
        if attr in self.__class__.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self.experiment, attr)

    def __dir__(self):
        return sorted(set(super().__dir__() + self.experiment.__dir__()))


if __name__ == '__main__':
    fire.Fire(EntryPoint)
