import torch

from dataclasses import dataclass, field
from typing import Optional, List, Any
from torch import Tensor

from fairseq import options, utils


def compute_SLM(
    self,
    src_tokens,
    **kwargs
):
    x, extra = compute_model(self.encoder, src_tokens)
    return x, extra


def compute_model(self, src_tokens, **kwargs):
    mask_idx = self.dictionary.index("<mask>")
    query_tokens = src_tokens.new_full(src_tokens.size(), mask_idx)

    features, _ = compute_encoder(self.sentence_encoder, src_tokens, query_tokens)
    logits = self.output_layer(features)
    return logits, {}


def compute_encoder(self, src_tokens, query_tokens=None, **kwargs):
    padding_mask = src_tokens.eq(self.padding_idx)

    def compute_embedding(tokens):
        x = self.embed_tokens(tokens)

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.embed_positions is not None:
            x = x + self.embed_positions(tokens)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        return x
    
    q = compute_embedding(query_tokens)
    c = compute_embedding(src_tokens)
    q_mask, f_mask, b_mask = make_slm_attention_mask(c)

    f, b = c, c

    for i, layer in enumerate(self.layers):
        q, f, b, _ = compute_transformer_encoder_layer(
            layer, q, f, b, q_mask, f_mask, b_mask, padding_mask,
        )
    
    q = q.transpose(0, 1)
    return q, {}


def merge(q, f, b):
    x = torch.cat((q, f, b), dim=1)
    return x


def split(x):
    sz = x.size(1) // 3
    q, f, b = x[:, :sz], x[:, sz:sz*2], x[:, sz*2:]
    return q, f, b


def compute_transformer_encoder_layer(
    self,
    q,
    f,
    b,
    q_mask: torch.Tensor = None,
    f_mask: torch.Tensor = None,
    b_mask: torch.Tensor = None,
    self_attn_padding_mask: torch.Tensor = None,
):
    def reuse_fn(x, residual):
        x = self.dropout_module(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x
   
    residual_q, residual_f, residual_b = q, f, b
    q, f, b, attn = triple_stream_self_attention(
        self.self_attn,
        q,
        f,
        b,
        query_mask=q_mask,
        forward_mask=f_mask,
        backward_mask=b_mask,
        key_padding_mask=self_attn_padding_mask,
    )
    q = reuse_fn(q, residual_q)
    f = reuse_fn(f, residual_f)
    b = reuse_fn(b, residual_b)
    return q, f, b, attn


def triple_stream_self_attention(
    self,
    q,
    f,
    b,
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
    return q, f, b, {}


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
        return torch.cat((fm, bm), 1).eq(1)

    query_mask = make_query_stream_mask(sz)
    forward_mask = make_forward_stream_mask(sz)
    backward_mask = make_backward_stream_mask(sz)

    return query_mask.to(tensor.device), forward_mask.to(tensor.device), backward_mask.to(tensor.device)