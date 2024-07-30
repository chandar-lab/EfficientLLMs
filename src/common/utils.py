import json
import _jsonnet
import os
from typing import Dict, Any
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
    if "weight_quantizer" in config.keys():
        config['model']['weight_quantize_module'] = config.pop('weight_quantizer')
    if "act_quantizer" in config.keys():
        config['model']['act_quantize_module'] = config.pop('act_quantizer')
    if "grad_quantizer" in config.keys():
        config['model']['grad_quantize_module'] = config.pop('grad_quantizer')

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


def load_and_create_experiment_name(config_files_names: str) -> (Dict[str, Any], str):
    """
    Generate a unique experiment name based on the provided configurations.
    The process involves removing unnecessary keys, unrolling nested configurations,
    and creating a hash from the resulting JSON string.

    Args:
        config_files_names (str): A comma-separated list of configuration file names.

    Returns:
        Tuple[Dict[str, Any], str]: The output dictionary containing experiment configurations and the unique experiment name.
    """
    # Extract and concatenate the base names without extensions
    config_names = '_'.join([os.path.splitext(os.path.basename(f.strip()))[0] for f in config_files_names.split(',') if 'evaluation_task' not in f])

    # Load and process the configurations
    config = load_configs(config_files_names)
    processed_config = remove_useless_key(copy.deepcopy(config))
    processed_config['evaluation_task'] = {}
    unrolled_config = unroll_configs(processed_config)

    # Generate a unique hash for the experiment name
    unrolled_json = json.dumps(unrolled_config, sort_keys=True)
    hash_name = hashlib.md5(unrolled_json.encode()).hexdigest()[:8]
    _unique_name = config.pop('_unique_name', False)
    exp_name = f"{config_names}_{hash_name}" if _unique_name else config_names

    return config, exp_name
