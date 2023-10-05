from transformers import GPT2Config
from flash_attn.models.gpt import GPTLMHeadModel
from typing import Optional, Dict
from models import Base_Model


@Base_Model.register("gpt2")
class CausalGPT2(GPT2LMHeadModel, Base_Model):
    def __init__(
            self,
            config: Optional[GPT2Config] = None,
            **kwargs,
    ):
        assert config is not None
        # super().__init__(config)
        GPT2LMHeadModel.__init__(self, config)
        Base_Model.__init__(self, **kwargs)


if __name__ == "__main__":
    from common import Params

    # DiT5Config.default_implementation = "t5"
    model = Base_Model.from_params(
        Params(
            {
                "type": "gpt2",
                "config": {
                    "activation_function": "gelu_new",
                    "attn_pdrop": 0.1,
                    "bos_token_id": 50256,
                    "embd_pdrop": 0.1,
                    "eos_token_id": 50256,
                    "initializer_range": 0.02,
                    "layer_norm_epsilon": 1e-05,
                    "model_type": "gpt2",
                    "n_embd": 768,
                    "n_head": 12,
                    "n_inner": None,
                    "n_layer": 12,
                    "n_positions": 1024,
                    "reorder_and_upcast_attn": False,
                    "resid_pdrop": 0.1,
                    "scale_attn_by_inverse_layer_idx": False,
                    "scale_attn_weights": True,
                    "summary_activation": None,
                    "summary_first_dropout": 0.1,
                    "summary_proj_to_labels": True,
                    "summary_type": "cls_index",
                    "summary_use_proj": True,
                    "transformers_version": "4.33.1",
                    "use_cache": True,
                    "vocab_size": 50257
                },
                "weight_quantize_module": {
                        "N_bits": 8,
                        "signed": 1,
                        "type": "lsq",
                        "use_grad_scaled": 1,
                        },
                "exp_name": "test_gpt2",
                "save_path": "./save",
                "device": "cuda",
            }
        )
    )
    print(model)
