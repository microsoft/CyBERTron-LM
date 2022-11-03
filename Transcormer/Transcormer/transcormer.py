import torch
from dataclasses import dataclass, field
from typing import Optional, List, Any

from fairseq import options, utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer_lm import (
    TransformerLanguageModelConfig,
    TransformerLanguageModel,
    base_lm_architecture,
    transformer_lm_gpt,
)
from torch import Tensor


@register_model("transcormer", dataclass=TransformerLanguageModelConfig)
class TranscormerLanguageModel(TransformerLanguageModel):

    def __init__(self, decoder):
        super().__init__(decoder)
        self.query = decoder.dictionary.add_symbol("<mask>")

    def forward(self, src_tokens, **kwargs):
        # Shape(x) : T x B x C
        query_tokens = src_tokens.new_full(
            src_tokens.size(), self.query
        )
        x, extra = _forward_decoder(
            self.decoder, src_tokens, query_tokens, **kwargs
        )
        x = self.output_layer(x)
        return x, extra

    def get_targets(self, sample, net_output):
        return sample['net_input']['src_tokens']

    @classmethod
    def build_model(cls, args, task):
        task.dictionary.add_symbol("<mask>")
        model = TransformerLanguageModel.build_model(args, task)
        return cls(model.decoder)


def _forward_decoder(
    self,
    src_tokens, 
    query_tokens,
    features_only: bool = False,
    alignment_layer: Optional[int] = None,
    alignment_heads: Optional[int] = None,
    return_all_hiddens: bool = False,
    **kwargs
):
    x = _forward_embedding(self, src_tokens)
    q = _forward_embedding(self, query_tokens)
    f, b = x, x
    q_mask, f_mask, b_mask = make_slm_attention_mask(x)

    self_attn_padding_mask: Optional[Tensor] = None
    if src_tokens.eq(self.padding_idx).any():
        self_attn_padding_mask = src_tokens.eq(self.padding_idx)

    # decoder layers
    attn: Optional[Tensor] = []
    inner_states: List[Optional[Tensor]] = [x]
    for idx, layer in enumerate(self.layers):
        q, f, b, layer_attn = _forward_single_decoder_layer(
            layer, q, f, b, 
            q_mask, f_mask, b_mask, self_attn_padding_mask,
        )
        inner_states.append(q)
        attn.append(layer_attn)

    if attn is not None:
        if alignment_heads is not None:
            attn = attn[:alignment_heads]

        if isinstance(attn, list):
            for i in range(len(attn)):
                attn[i]['f_attn'] = attn[i]['f_attn'].mean(dim=0)
                attn[i]['b_attn'] = attn[i]['b_attn'].mean(dim=0)
                attn[i]['q_attn'] = attn[i]['q_attn'].mean(dim=0)
        else:
            # average probabilities over heads
            attn = attn.mean(dim=0)
    
    if self.layer_norm is not None:
        q = self.layer_norm(q)
    
    q = q.transpose(0, 1)
    return q, {'attn': attn}


def _forward_embedding(self, src_tokens):
    positions = (
        self.embed_positions(src_tokens)
        if self.embed_positions is not None
        else None
    )

    x = self.embed_scale * self.embed_tokens(src_tokens)

    if self.project_in_dim is not None:
        x = self.project_in_dim(x)

    if positions is not None:
        x += positions

    if self.layernorm_embedding is not None:
        x = self.layernorm_embedding(x)

    x = self.dropout_module(x)

    # B x T x C => T x B x C
    x = x.transpose(0, 1)
    return x


def _forward_single_decoder_layer(
    self,
    q,
    f,
    b,
    query_mask: Optional[torch.Tensor] = None,
    forward_mask: Optional[torch.Tensor] = None,
    backward_mask: Optional[torch.Tensor] = None,
    self_attn_padding_mask: Optional[torch.Tensor] = None,
):
    def reuse_fn(x, residual):
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        x = maybe_layer_norm(self, self.self_attn_layer_norm, x, after=True)
        
        residual = x
        x = maybe_layer_norm(self, self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        x = maybe_layer_norm(self, self.final_layer_norm, x, after=True)
        return x

    residual_q, residual_f, residual_b = q, f, b

    q = maybe_layer_norm(self, self.self_attn_layer_norm, q, before=True)
    f = maybe_layer_norm(self, self.self_attn_layer_norm, f, before=True)
    b = maybe_layer_norm(self, self.self_attn_layer_norm, b, before=True)

    q, f, b, attns = triple_stream_self_attention(
        self.self_attn, q, f, b, 
        query_mask=query_mask,
        forward_mask=forward_mask,
        backward_mask=backward_mask,
        key_padding_mask=self_attn_padding_mask,
    )

    q = reuse_fn(q, residual_q)
    f = reuse_fn(f, residual_f)
    b = reuse_fn(b, residual_b)
    return q, f, b, attns


def triple_stream_self_attention(
    self,
    q: torch.Tensor,
    f: torch.Tensor,
    b: torch.Tensor,
    query_mask: torch.Tensor = None,
    forward_mask: torch.Tensor = None,
    backward_mask: torch.Tensor = None,
    key_padding_mask: torch.Tensor = None,
):

    bsz, embed_dim = q.size(1), q.size(2)

    def transpose_fn(x):
        return x.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def fill_mask(attn_weights, attn_mask):
        return attn_weights.masked_fill(
            attn_mask.unsqueeze(0),
            float('-inf')
        )

    def attn_fn(_q, k, v, mask=None, padding_mask=None, is_content=False):
        _q = transpose_fn(self.scaling * self.q_proj(_q))
        attn_weights = torch.bmm(_q, k.transpose(1, 2))
        
        if mask is not None:
            attn_weights = fill_mask(attn_weights, mask)

        q_len, k_len = _q.size(1), k.size(1)

        if padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, q_len, k_len)

            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            if is_content:
                padding_mask = padding_mask & torch.eye(q_len).ne(1).to(padding_mask)

            attn_weights = attn_weights.masked_fill(
                padding_mask,
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, q_len, k_len)

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1,  
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_weights = self.dropout_module(attn_weights)
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        return self.out_proj(attn), attn_weights

    def get_key_and_value(x):
        """
            k: [B * H, T, C]
            v: [B * H, T, C]
        """
        k = transpose_fn(self.k_proj(x))
        v = transpose_fn(self.v_proj(x))
        return k, v

    fk, fv = get_key_and_value(f)
    bk, bv = get_key_and_value(b)
    
    qk = torch.cat([fk, bk], dim=1)
    qv = torch.cat([fv, bv], dim=1)

    f, f_attn = attn_fn(f, fk, fv, forward_mask, key_padding_mask, True)
    b, b_attn = attn_fn(b, bk, bv, backward_mask, key_padding_mask, True)
    q_key_padding_mask = None if key_padding_mask is None else \
        torch.cat([key_padding_mask, key_padding_mask], dim=1)
    q, q_attn = attn_fn(q, qk, qv, query_mask, q_key_padding_mask)
    return q, f, b, {"f_attn": f_attn, "b_attn": b_attn, "q_attn": q_attn}


def make_slm_attention_mask(tensor):
    # Please note we need to each token to attend at least one token (including padding token)
    sz = tensor.size(0)

    def make_forward_stream_mask(l):
        return torch.triu(torch.ones((l, l)), 1).eq(1)

    def make_backward_stream_mask(l):
        return torch.tril(torch.ones((l, l)), -1).eq(1)

    def make_query_stream_mask(l):
        fm = torch.triu(torch.ones((l, l)), 0)
        bm = torch.tril(torch.ones((l, l)), 0)
        bm = torch.tril(torch.ones((l, l)), 0)
        return torch.cat((fm, bm), 1).eq(1)

    query_mask = make_query_stream_mask(sz)
    forward_mask = make_forward_stream_mask(sz)
    backward_mask = make_backward_stream_mask(sz)

    return query_mask.to(tensor.device), forward_mask.to(tensor.device), backward_mask.to(tensor.device)


def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
    assert before ^ after
    if after ^ self.normalize_before:
        return layer_norm(x)
    else:
        return x


@register_model_architecture("transcormer", "transcormer")
def transcormer(args):
    base_lm_architecture(args)


@register_model_architecture("transcormer", "transcormer_gpt")
def transcormer_gpt(args):
    transformer_lm_gpt(args)


@register_model_architecture("transcormer", "transcormer_gpt_small")
def transcormer_gpt_small(args):
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    transcormer_gpt(args)