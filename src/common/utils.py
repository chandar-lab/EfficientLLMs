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


def create_quant_config_name(w_cfg: dict, a_cfg: dict, g_cfg: dict = None, s_one_cfg: dict = None,
                             s_two_cfg: dict = None):
    """

    Args:
        w_cfg: quantization config for weights
        a_cfg: quantization config for activations
        g_cfg: quantization config for gradients
        s_one_cfg: quantization config for first optimizer states
        s_two_cfg: quantization config for second optimizer states

    Returns:

    """
    def format_quantizer_info(cfg, name_prefix):
        bit = cfg['N_bits']
        g = '_' + str(cfg['granularity']).replace('-', '_')
        qtype = '' if cfg['type'] == 'normal' else '_' + str(cfg['type']).replace('-', '_')
        sym = '_sym' if cfg['symmetric'] else '_asym'
        return f'{name_prefix}{bit}{qtype}{sym}{g}'

    def format_quantizer_info_multi(cfg1, name_prefix1, cfg2, name_prefix2):
        bit1 = cfg1['N_bits']
        bit2 = cfg2['N_bits']
        sym = '_sym' if cfg2['symmetric'] else '_asym'
        first_part = f'{name_prefix1}{bit1}{name_prefix2}{bit2}{sym}'
        g1 = '_' + str(cfg1['granularity']).replace('-', '_')
        g2 = '_' + str(cfg2['granularity']).replace('-', '_')
        qtype = '' if cfg1['type'] == 'normal' else '_' + str(cfg1['type']).replace('-', '_')
        return f'{first_part}{qtype}{g1}{g2}'

    use_weight_quantizer = bool(w_cfg)
    use_act_quantizer = bool(a_cfg)

    if use_weight_quantizer and use_act_quantizer:
        config_name = format_quantizer_info_multi(w_cfg, 'W', a_cfg, 'A')
    elif use_weight_quantizer:
        config_name = format_quantizer_info(w_cfg, 'W')
    elif use_act_quantizer:
        config_name = format_quantizer_info(a_cfg, 'A')
    else:
        config_name = ""

    prefix = '' if config_name == "" else '_'
    if bool(g_cfg):
        config_name += prefix + format_quantizer_info(g_cfg, 'G')

    use_first_state_quantizer = bool(s_one_cfg)
    use_second_state_quantizer = bool(s_two_cfg)

    if use_first_state_quantizer and use_second_state_quantizer:
        config_name += prefix + format_quantizer_info_multi(s_one_cfg, 'S1_', s_two_cfg, 'S2_')
    elif use_first_state_quantizer:
        config_name += prefix + format_quantizer_info(s_one_cfg, 'S1_')
    elif use_second_state_quantizer:
        config_name += prefix + format_quantizer_info(s_two_cfg, 'S2_')
    else:
        config_name += ""

    return config_name


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
    _config['evaluation_task'] = {}
    model_arch = _config['model']['type']
    if _config['optimizer']:
        quantizer_cfg_name = create_quant_config_name(_config['model']['weight_quantize_module'],
                                                      _config['model']['act_quantize_module'],
                                                      _config['model']['grad_quantize_module'],
                                                      _config['optimizer']['first_state_quantize_module'],
                                                      _config['optimizer']['second_state_quantize_module'])
    else:
        quantizer_cfg_name = create_quant_config_name(_config['model']['weight_quantize_module'],
                                                      _config['model']['act_quantize_module'],
                                                      _config['model']['grad_quantize_module'],)
    _config = unroll_configs(_config)
    # Convert the unrolled dictionary to a JSON string and hash it
    unrolled_json = json.dumps(_config, sort_keys=True)
    hash_name = hashlib.md5(unrolled_json.encode()).hexdigest()[:8]
    exp_name = f"{model_arch}_{quantizer_cfg_name}_{hash_name}"
    return exp_name
