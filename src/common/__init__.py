# for the dependency injection using Jsonnet files
# code from: https://github.com/kazemnejad/pt_hf_base

from enum import Enum
from typing import Dict, Any

from .from_params import FromParams, ConfigurationError
from .lazy import Lazy
from .params import Params
from .registrable import Registrable
from .utils import *

assert FromParams
assert Lazy
assert Params
assert Registrable
assert ConfigurationError

JsonDict = Dict[str, Any]
