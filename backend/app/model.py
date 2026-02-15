from pathlib import Path
import re

import torch
import torch.nn as nn
from torch import Tensor

from .config import Settings


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * torch.log(torch.tensor(10000.0)) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(1)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        return self.dropout(token_embedding + self.pos_embedding[: token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens.long())


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.batch_first = False
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ) -> Tensor:
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor) -> Tensor:
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


class LegacyEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.batch_first = True
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, src: Tensor, src_key_padding_mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len = src.shape
        positions = torch.arange(seq_len, device=src.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.word_embedding(src) + self.position_embedding(positions)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class LegacyDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.batch_first = True
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor,
        memory_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, seq_len = tgt.shape
        positions = torch.arange(seq_len, device=tgt.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.word_embedding(tgt) + self.position_embedding(positions)
        out = self.decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.fc_out(out)


class LegacySeq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.batch_first = True
        self.pad_id = 0
        self.encoder = LegacyEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )
        self.decoder = LegacyDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )

    @property
    def generator(self) -> nn.Module:
        return nn.Identity()

    def make_src_key_padding_mask(self, src: Tensor) -> Tensor:
        return src == self.pad_id

    def make_tgt_key_padding_mask(self, tgt: Tensor) -> Tensor:
        return tgt == self.pad_id

    def make_tgt_causal_mask(self, tgt_len: int, device: torch.device) -> Tensor:
        return generate_square_subsequent_mask(tgt_len, device)

    def encode(self, src: Tensor, src_key_padding_mask: Tensor | None = None) -> Tensor:
        return self.encoder(src, src_key_padding_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor,
        memory_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        return self.decoder(tgt, memory, tgt_mask, memory_key_padding_mask, tgt_key_padding_mask)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        src_key_padding_mask = self.make_src_key_padding_mask(src)
        tgt_key_padding_mask = self.make_tgt_key_padding_mask(tgt)
        tgt_mask = self.make_tgt_causal_mask(tgt.size(1), tgt.device)

        memory = self.encode(src, src_key_padding_mask)
        decoder_out = self.decode(
            tgt,
            memory,
            tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.generator(decoder_out)


def generate_square_subsequent_mask(size: int, device: torch.device) -> Tensor:
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)


def load_transformer_model(settings: Settings, vocab_size: int) -> Seq2SeqTransformer:
    model_path = settings.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    device = torch.device(settings.device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise RuntimeError("Invalid checkpoint format. Expected a state_dict or checkpoint dictionary.")

    legacy_layout = any(k.startswith("encoder.word_embedding.weight") for k in state_dict.keys())

    if legacy_layout:
        enc_layer_indices = [
            int(m.group(1))
            for k in state_dict.keys()
            for m in [re.match(r"encoder\.encoder\.layers\.(\d+)\.", k)]
            if m is not None
        ]
        dec_layer_indices = [
            int(m.group(1))
            for k in state_dict.keys()
            for m in [re.match(r"decoder\.decoder\.layers\.(\d+)\.", k)]
            if m is not None
        ]

        inferred_enc_layers = (max(enc_layer_indices) + 1) if enc_layer_indices else settings.num_encoder_layers
        inferred_dec_layers = (max(dec_layer_indices) + 1) if dec_layer_indices else settings.num_decoder_layers

        inferred_d_model = state_dict["encoder.word_embedding.weight"].shape[1]
        inferred_vocab = state_dict["encoder.word_embedding.weight"].shape[0]
        inferred_max_len = state_dict["encoder.position_embedding.weight"].shape[0]

        ff_weight = state_dict.get("encoder.encoder.layers.0.linear1.weight")
        inferred_dim_ff = ff_weight.shape[0] if ff_weight is not None else settings.dim_feedforward

        model = LegacySeq2SeqTransformer(
            vocab_size=inferred_vocab or vocab_size,
            d_model=inferred_d_model,
            nhead=settings.nhead,
            num_encoder_layers=inferred_enc_layers,
            num_decoder_layers=inferred_dec_layers,
            dim_feedforward=inferred_dim_ff,
            dropout=settings.dropout,
            max_len=inferred_max_len,
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
    else:
        model = Seq2SeqTransformer(
            num_encoder_layers=settings.num_encoder_layers,
            num_decoder_layers=settings.num_decoder_layers,
            emb_size=settings.d_model,
            nhead=settings.nhead,
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            dim_feedforward=settings.dim_feedforward,
            dropout=settings.dropout,
        ).to(device)
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model
