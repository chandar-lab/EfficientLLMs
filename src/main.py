import copy
from runtime import Runtime
from common import Params, load_and_create_experiment_name
import fire


class EntryPoint(object):
    experiment = None
    configs = None

    def __init__(self, configs: str = "../configs/gpt2_w8a8g8_per_token.jsonnet"):

        _config, exp_name = load_and_create_experiment_name(configs)
        _config_copy = copy.deepcopy(_config)
        self.experiment = Runtime.from_params(Params(_config), exp_name=exp_name, _config_copy=_config_copy)

    def __getattr__(self, attr):
        if attr in self.__class__.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self.experiment, attr)

    def __dir__(self):
        return sorted(set(super().__dir__() + self.experiment.__dir__()))


if __name__ == '__main__':
    fire.Fire(EntryPoint)
