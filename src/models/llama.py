try:
    from flash_attn.models.gpt import GPTLMHeadModel
    from flash_attn.models.llama import llama_config_to_gpt2_config, inv_remap_state_dict_hf_llama
except ImportError:
    from transformers.models.gpt2 import GPT2LMHeadModel as GPTLMHeadModel

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None

from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import LlamaForCausalLM, LlamaConfig
from typing import Optional, Dict
from collections import OrderedDict, namedtuple
from transformers import GPT2Config
from models import Base_Model
from common import FromParams
import torch
from torch.nn import CrossEntropyLoss


class MyLlamaConfig(LlamaConfig, FromParams):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@Base_Model.register("llama")
class CausalLlama(GPTLMHeadModel, Base_Model):
    def __init__(
            self,
            config: Optional[MyLlamaConfig] = None,
            use_flash_attn: bool = True,
            fused_bias_fc: bool = True,
            fused_mlp: bool = True,
            fused_dropout_add_ln: bool = True,
            residual_in_fp32: bool = True,
            pad_vocab_size_multiple: int = 8,
            **kwargs,

    ):
        # for optimized gpt2
        config = llama_config_to_gpt2_config(config)
        config.use_flash_attn = use_flash_attn
        config.fused_bias_fc = fused_bias_fc
        config.fused_mlp = fused_mlp
        config.fused_dropout_add_ln = fused_dropout_add_ln
        config.residual_in_fp32 = residual_in_fp32
        config.pad_vocab_size_multiple = pad_vocab_size_multiple
        GPTLMHeadModel.__init__(self, config)
        Base_Model.__init__(self, **kwargs)

    #  here we ignore attention_mask to make it compatible with HF trainer. The MHA in flash-attention should
    #  be reimplement and integrate attention_mask like here:
    #  https://github.com/huggingface/transformers/blob/0864dd3beb238b7bec3528a3d1d6c17a28f51a51/src/transformers/models/llama/modeling_llama.py#L536
    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0,
                attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                return_dict: Optional[bool] = None,
                ):
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

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
        )
