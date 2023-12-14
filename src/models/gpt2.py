try:
    from flash_attn.models.gpt import GPTLMHeadModel
except ImportError:
    from transformers.models.gpt2 import GPT2LMHeadModel as GPTLMHeadModel

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None

from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Optional, Dict
from collections import OrderedDict, namedtuple
from transformers import GPT2Config
from models import Base_Model
from common import FromParams
import torch
from torch.nn import CrossEntropyLoss


class MyGPT2Config(GPT2Config, FromParams):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

@Base_Model.register("gpt2-hf")
class CausalGPT2_HF(GPT2LMHeadModel, Base_Model):
    def __init__(
            self,
            config: Optional[MyGPT2Config] = None,
            **kwargs,
    ):
        assert config is not None
        # for optimized gpt2
        GPT2LMHeadModel.__init__(self, config)
        Base_Model.__init__(self, **kwargs)

@Base_Model.register("gpt2")
class CausalGPT2(GPTLMHeadModel, Base_Model):
    def __init__(
            self,
            config: Optional[MyGPT2Config] = None,
            use_flash_attn: bool = True,
            fused_bias_fc: bool = True,
            fused_mlp: bool = True,
            fused_dropout_add_ln: bool = True,
            residual_in_fp32: bool = True,
            pad_vocab_size_multiple: int = 8,
            **kwargs,
    ):
        assert config is not None
        # for optimized gpt2
        config.use_flash_attn = use_flash_attn
        config.fused_bias_fc = fused_bias_fc
        config.fused_mlp = fused_mlp
        config.fused_dropout_add_ln = fused_dropout_add_ln
        config.residual_in_fp32 = residual_in_fp32
        config.pad_vocab_size_multiple = pad_vocab_size_multiple
        GPTLMHeadModel.__init__(self, config)
        Base_Model.__init__(self, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0,
                attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                return_dict: Optional[bool] = None,
                ):
        """
        input_ids: (batch, seqlen) int tensor
        inference_params: for generation. Adapted from Megatron-LM (and Apex)
        https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        assert (
                input_ids.ndim == 2
        ), f"Expected `input_ids` to have shape [b, slen], but got shape {input_ids.shape}"
        b, slen = input_ids.shape
        hidden_states = self.transformer(
            input_ids, position_ids=position_ids, inference_params=inference_params
        )
        if inference_params is not None:
            assert hidden_states.ndim == 3, "sequence_parallel is not supported in generation mode"
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        # # During inference, we want the full logit for sampling
        # if isinstance(self.lm_head, ColumnParallelLinear) and inference_params is not None:
        #     lm_logits, _ = all_gather_raw(lm_logits, self.lm_head.process_group)
        #     lm_logits = rearrange(lm_logits, "(n b) ... d -> b ... (n d)", b=b)
        # CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        # return CausalLMOutput(logits=lm_logits)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # if not return_dict:
        #     output = (lm_logits,) + transformer_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
        )


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
            }
        )
    )
    print(model)


"""
For gp2-FlashAttn mode:

model.transformer.layers[i] --> flash_attn.modules.block.Block
    mixer --> MHA
        mixer.Wqkv --> FusedDense
        mixer.inner_attn --> FlashSelfAttention
        mixer.inner_cross_attn --> FlashCrossAttention
        mixer.out_proj --> FusedDense
    norm1 --> LayerNorm
    mlp --> FusedMLP
        mlp.fc1 --> Linear
        mlp.fc2 --> Linear
    norm2 --> LayerNorm
    
The problem is FusedDense layer is only implemented for bf16 or fp16. So for quantizing weight it's better to 
set `fused_bias_fc` to False

"""