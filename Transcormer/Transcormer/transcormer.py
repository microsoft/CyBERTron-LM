# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional

import torch
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta import RobertaModel, roberta_base_architecture


@register_model("transcormer")
class TranscormerModel(RobertaModel):
    """
        TranscormerModel inherited from RoBERTa, with a prediction head during pre-training:
        a `sliding language modeling` head
    """
    def __init__(self, args, encoder):
        super().__init__(args, encoder)
        self.mask_idx = self.encoder.dictionary.add_symbol("<mask>")

    def compute(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        **kwargs
    ):
        """ Function for computing sliding language modeling (SLM). """

        # Shape(x) : T x B x C
        query_tokens = src_tokens.new_full(
            src_tokens.size(), self.mask_idx
        )
        x, extra = extract_encoder_features(
            self.encoder,
            src_tokens,
            query_tokens,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
            **kwargs
        )
        return x, extra


def extract_encoder_features(
    self,
    src_tokens,
    query_tokens,
    features_only: bool = False,
    return_all_hiddens: bool = False,
    **unused
) -> tuple:
    """ Predict logits from Transcormer encoder features"""
    inner_states, attns = extract_sentence_encoder_features(
        self.sentence_encoder,
        src_tokens,
        query_tokens,
        last_state_only=not return_all_hiddens,
        **unused
    )
    features = inner_states[-1].transpose(0, 1)
    if not features_only:
        x = self.output_layer(features)
    return x, {"inner_states": inner_states if return_all_hiddens else None, "attns": attns}


def extract_sentence_encoder_features(
    self,
    src_tokens,
    query_tokens,
    last_state_only: bool = False,
    **unused
):
    """ Extract features from Transcormer Encoder """
    padding_mask = src_tokens.eq(self.padding_idx)

    w = extract_word_embedding(self, src_tokens)
    q = extract_word_embedding(self, query_tokens)
    f, b = w, w

    query_mask, forward_mask, backward_mask = make_slm_attention_mask(w)

    attns = {'q_attn': [], 'f_attn': [], 'b_attn': []}
    inner_states = []

    if not last_state_only:
        inner_states.append(q)

    for layer in self.layers:
        q, f, b, attn = calculate_transcormer_layer_states(
            layer, q, f, b, query_mask, forward_mask, backward_mask, padding_mask,
        )
        attns['q_attn'].append(attn[0])
        attns['f_attn'].append(attn[1])
        attns['b_attn'].append(attn[2])

        if not last_state_only:
            inner_states.append(q)

    if last_state_only:
        inner_states = [q]

    return inner_states, attns


def extract_word_embedding(self, tokens, padding_mask=None):
    """ Extract word embedding from sentence encoder """

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

    # B x T x C => T x B x C
    x = x.transpose(0, 1)
    return x


def calculate_transcormer_layer_states(
    self,
    q,
    f,
    b,
    query_mask: Optional[torch.Tensor] = None,
    forward_mask: Optional[torch.Tensor] = None,
    backward_mask: Optional[torch.Tensor] = None,
    self_attn_padding_mask: Optional[torch.Tensor] = None,
):
    """ Calculate feature states of transcormer layer """
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
    """ Triple-Stream Self-Attention for calcuating sliding language modeling """
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
    return q, f, b, (q_attn, f_attn, b_attn)


def make_slm_attention_mask(x):
    """ Generate query/forward/backward attention mask for sliding language modeling """

    # Please note we need to each token to attend at least one token (including padding token)
    sz = x.size(0)

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

    return query_mask.to(x.device), forward_mask.to(x.device), backward_mask.to(x.device)


@register_model_architecture("transcormer", "transcormer")
def transcormer(args):
    """ Parameters used in RoBERTa base (Y Liu et al, 2021) """
    roberta_base_architecture(args)


@register_model_architecture("transcormer", "transcormer_small")
def transcormer_small(args):
    """ Parameters used in a smaller setting """
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    transcormer(args)
