import json
import _jsonnet
from common import Params
from typing import Dict, Any, List
import copy
import hashlib

def load_configs(config_files_names: str) -> Dict[str, Any]:
    """
    Merge and load configurations from multiple files into a single dictionary.

    Args:
        config_files_names (str): A comma-separated list of configuration file names.

    Returns:
        Dict[str, Any]: The output dictionary containing experiment configurations.
    """
    filenames = [f.strip() for f in config_files_names.split(",")]
    jsonnet_str = "+".join([f'(import "{f}")' for f in filenames])
    json_str = _jsonnet.evaluate_snippet("snippet", jsonnet_str)
    config: Dict[str, Any] = json.loads(json_str)
    if "quantizer" in config.keys():
        config['model']['weight_quantize_module'] = config.pop('quantizer')

    return config

def remove_useless_key(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively removes all keys starting with '_' from a dictionary. These keys typically represent
    configuration options that are not critical for training or evaluation and can be considered as
    non-essential or internal.

    Args:
        cfg (Dict[str, Any]): The input dictionary containing configuration options.

    Returns:
        Dict[str, Any]: The output dictionary with the unnecessary keys removed.
    """
    cleaned_dict = {}
    for key, value in cfg.items():
        if not key.startswith('_'):
            if isinstance(value, dict):
                cleaned_dict[key] = remove_useless_key(value)
            else:
                cleaned_dict[key] = value
    return cleaned_dict


def unroll_configs(cfg: Dict[str, Any], parent_key='', sep='_') -> Dict[str, Any]:
    """
    Recursively unroll a nested dictionary of configurations and remove keys with None values.

    Args:
        cfg (Dict[str, Any]): The input dictionary containing configuration options.
        parent_key (str): The parent key for the current level of recursion.
        sep (str): The separator used to separate parent and child keys.

    Returns:
        Dict[str, Any]: The output unrolled dictionary.
    """
    items = {}
    for key, value in cfg.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(unroll_configs(value, new_key, sep=sep))
        elif value is not None:  # Exclude keys with None values
            items[new_key] = value
    return items

def creat_unique_experiment_name(config: Dict[str, Any]) -> str:
    """
    Generate a unique experiment name based on the provided configurations.
    The process involves removing unnecessary keys, unrolling nested configurations,
    and creating a hash from the resulting JSON string.

    Args:
        config (Dict[str, Any]): The input dictionary containing experiment configurations.

    Returns:
        str: A unique experiment name.
    """
    _config = copy.deepcopy(config)
    _config = remove_useless_key(_config)
    model_arch = _config['model']['type']
    use_quantizer = len(_config['model']['weight_quantize_module'].keys())>0
    _config = unroll_configs(_config)
    # Convert the unrolled dictionary to a JSON string and hash it
    unrolled_json = json.dumps(_config, sort_keys=True)
    hash_name = hashlib.md5(unrolled_json.encode()).hexdigest()[:8]
    if use_quantizer:
        exp_name = f"{model_arch}_{_config['model_weight_quantize_module_N_bits']}bit_{_config['model_weight_quantize_module_type']}_{hash_name}"
    else:
        exp_name = f"{model_arch}_{hash_name}"
    return exp_name
