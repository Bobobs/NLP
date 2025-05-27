from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.marian.configuration_marian import MarianConfig

# ─────────────────────────────────────────────────────────────────────────────
# Embeddings & positional encoding
# ─────────────────────────────────────────────────────────────────────────────
class MarianEmbeddings(nn.Module):
    """Token + positional embeddings shared by encoder & decoder."""

    def __init__(self, config: MarianConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.LayerNorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

        x = self.word_embeddings(input_ids) + self.position_embeddings(positions)
        return self.dropout(self.LayerNorm(x))


# ─────────────────────────────────────────────────────────────────────────────
# Attention & FFN building blocks (identical to BART)
# ─────────────────────────────────────────────────────────────────────────────
class MarianAttention(nn.Module):
    def __init__(self, config: MarianConfig, is_cross_attention: bool = False):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads if not is_cross_attention else config.decoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(config.attention_dropout)

    def _shape(self, tensor: torch.Tensor, bsz: int, seq_len: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.size()

        # Self‑attention = queries, keys, values all from hidden_states
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(key_value_states if key_value_states is not None else hidden_states)
        value_states = self.v_proj(key_value_states if key_value_states is not None else hidden_states)

        query_states = self._shape(query_states, bsz, seq_len)
        key_states = self._shape(key_states, bsz, -1)
        value_states = self._shape(value_states, bsz, -1)

        attn_weights = torch.einsum("bnqd,bnkd->bnqk", query_states, key_states)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask  # masked positions → -inf

        attn_scores = torch.softmax(attn_weights, dim=-1)
        attn_scores = self.dropout(attn_scores)

        attn_output = torch.einsum("bnqk,bnvd->bnqv", attn_scores, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        return self.out_proj(attn_output), attn_scores


class MarianFeedForward(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.activation_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


# ─────────────────────────────────────────────────────────────────────────────
# Encoder & Decoder layers
# ─────────────────────────────────────────────────────────────────────────────
class MarianEncoderLayer(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        self.self_attn = MarianAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = MarianFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, attention_mask):
        attn_output, _ = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = hidden_states + self.dropout(attn_output)
        hidden_states = self.self_attn_layer_norm(hidden_states)

        ffn_output = self.ffn(hidden_states)
        hidden_states = hidden_states + self.dropout(ffn_output)
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class MarianDecoderLayer(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        self.self_attn = MarianAttention(config)
        self.cross_attn = MarianAttention(config, is_cross_attention=True)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.cross_attn_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = MarianFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, encoder_states, self_mask, cross_mask):
        self_attn_out, _ = self.self_attn(hidden_states, attention_mask=self_mask)
        hidden_states = hidden_states + self.dropout(self_attn_out)
        hidden_states = self.self_attn_layer_norm(hidden_states)

        cross_attn_out, _ = self.cross_attn(
            hidden_states, key_value_states=encoder_states, attention_mask=cross_mask
        )
        hidden_states = hidden_states + self.dropout(cross_attn_out)
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        ffn_out = self.ffn(hidden_states)
        hidden_states = hidden_states + self.dropout(ffn_out)
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


# ─────────────────────────────────────────────────────────────────────────────
# Stacks
# ─────────────────────────────────────────────────────────────────────────────
class MarianEncoder(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        self.embed = MarianEmbeddings(config)
        self.layers = nn.ModuleList([MarianEncoderLayer(config) for _ in range(config.encoder_layers)])

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class MarianDecoder(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        self.embed = MarianEmbeddings(config)
        self.layers = nn.ModuleList([MarianDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.linear = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, encoder_states, self_mask, cross_mask):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, encoder_states, self_mask, cross_mask)
        return self.linear(x)


# ─────────────────────────────────────────────────────────────────────────────
# The full sequence‑to‑sequence model
# ─────────────────────────────────────────────────────────────────────────────
class MarianModel(PreTrainedModel):
    config_class = MarianConfig
    base_model_prefix = "model"

    def __init__(self, config: MarianConfig):
        super().__init__(config)
        self.encoder = MarianEncoder(config)
        self.decoder = MarianDecoder(config)

        # Share embedding weights == weight tying (encoder & decoder + output layer)
        self.decoder.embed.word_embeddings.weight = self.encoder.embed.word_embeddings.weight
        self.decoder.linear.weight = self.encoder.embed.word_embeddings.weight

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> BaseModelOutput:
        enc_states = self.encoder(input_ids, attention_mask)
        logits      = self.decoder(decoder_input_ids, enc_states, decoder_attention_mask, attention_mask)
        return BaseModelOutput(last_hidden_state=logits, hidden_states=None, attentions=None)


class MarianMTModel(MarianModel):
    """Marian model **with generation helpers** (beam‑search etc.)."""

    def __init__(self, config: MarianConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Seq2SeqLMOutput:
        if decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)

        outputs = super().forward(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=None,
        )
        logits = outputs.last_hidden_state

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=None,
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )

    # ---- helper for teacher‑forcing ----
    def _shift_right(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        pad_token = self.config.pad_token_id
        decoder_start_token = self.config.decoder_start_token_id
        shifted = torch.zeros_like(input_ids)
        shifted[:, 0] = decoder_start_token
        shifted[:, 1:] = input_ids[:, :-1]
        shifted[input_ids == pad_token] = pad_token
        return shifted

# End of trimmed source
