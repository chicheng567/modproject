import torch
import torch.nn as nn
from transformers.utils import ModelOutput, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from transformers.utils.deprecation import deprecate_kwarg
from dataclasses import dataclass
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import *
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Optional, Tuple, Union, List
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.generation.logits_process import TopKLogitsWarper
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaDecoderLayer, LlamaRotaryEmbedding
import copy
class LlamaTextModelHeadless(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        skip_layers: Optional[int] = 0,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if skip_layers > 0:
            assert inputs_embeds is not None, "inputs_embeds must be provided when skip_layers > 0"
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )
        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers[skip_layers: self.config.num_hidden_layers]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

class LlamaForCausalLMHeadless(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaTextModelHeadless(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.top_k_warper = TopKLogitsWarper(top_k=config.top_k)

    def _sample(self, logits, temperature=1.0):
        logits = self.top_k_warper(None, logits) / temperature
        prob = nn.functional.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(prob, num_samples=1).squeeze(1)
        return next_tokens
    @torch.no_grad()
    def forward(
        self,
        skip_layers: Optional[int] = 0,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, #should not be None when auto_regressive is True
        past_key_values: Optional[Cache] = None, #shape:[(1, heads, seq_len, head_dim)] * num_layers
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None, #not used in DynamicCache, but used for compute current token positions.
        logits_to_keep: Union[int, torch.Tensor] = 1, #how many new tokens to generate, should be equal to cache position
    ) -> Tuple[torch.LongTensor, Cache]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if skip_layers > 0:
            assert inputs_embeds is not None, "inputs_embeds must be provided when skip_layers > 0"
        assert attention_mask is not None, "attention_mask must be provided"
        assert position_ids is not None, "position_ids must be provided"
        assert cache_position is not None, "cache_position must be provided"
        outputs: BaseModelOutputWithPast = self.model(
            skip_layers=skip_layers,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position
        )
        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = logits.view(-1, self.vocab_size)
        

        output_ids = self._sample(logits, self.config.temperature)
        return output_ids, past_key_values

class LlamaHead(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.first_layer = LlamaDecoderLayer(config, layer_idx=0)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        
    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        assert input_ids is not None, "input_ids must be provided"
        assert position_ids is not None, "position_ids must be provided"
        assert cache_position is not None, "cache_position must be provided"
        inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )
        
        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        layer_outputs = self.first_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            past_key_value=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **flash_attn_kwargs,
        )
        hidden_states = layer_outputs[0]
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )