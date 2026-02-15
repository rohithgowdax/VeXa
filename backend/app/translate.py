import re
from dataclasses import dataclass

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import nn

from .config import Settings
from .model import generate_square_subsequent_mask, load_transformer_model


@dataclass
class RuntimeResources:
    translator: "Translator"


class Translator:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: spm.SentencePieceProcessor,
        settings: Settings,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        self.device = torch.device(settings.device)

    @torch.inference_mode()
    def translate_text(self, text: str) -> str:
        normalized = " ".join(text.strip().split())
        if not normalized:
            return ""

        # Training context uses plain source ids (no BOS/EOS), with decoder starting at BOS.
        src_ids = self.tokenizer.encode(normalized, out_type=int)
        if not src_ids:
            return ""

        if self.settings.beam_size > 1:
            output_ids = self._beam_decode(src_ids)
        else:
            output_ids = self._greedy_decode(src_ids)

        translation = self.tokenizer.decode(output_ids) if output_ids else ""
        translation = re.sub(r"\s+", " ", translation).strip()
        return translation

    def _clean_output_ids(self, token_ids: list[int]) -> list[int]:
        cleaned: list[int] = []
        for tok in token_ids:
            if tok == self.settings.eos_id:
                break
            if tok in (self.settings.sos_id, self.settings.pad_id):
                continue
            cleaned.append(tok)
        return cleaned

    def _greedy_decode(self, src_ids: list[int]) -> list[int]:
        batch_first = bool(getattr(self.model, "batch_first", False))

        if batch_first:
            src = torch.tensor([src_ids], dtype=torch.long, device=self.device)
            ys = torch.tensor([[self.settings.sos_id]], dtype=torch.long, device=self.device)

            for _ in range(self.settings.max_len - 1):
                logits = self.model(src, ys)
                next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long, device=self.device)], dim=1)
                if next_token == self.settings.eos_id:
                    break

            return self._clean_output_ids(ys.squeeze(0).tolist())

        src = torch.tensor(src_ids, dtype=torch.long, device=self.device).unsqueeze(1)
        src_mask = torch.zeros((src.size(0), src.size(0)), device=self.device, dtype=torch.bool)
        memory = self.model.encode(src, src_mask)

        ys = torch.tensor([[self.settings.sos_id]], dtype=torch.long, device=self.device)
        for _ in range(self.settings.max_len - 1):
            tgt_mask = generate_square_subsequent_mask(ys.size(0), self.device)
            decoded = self.model.decode(ys, memory, tgt_mask)
            if decoded.dim() == 3 and decoded.size(0) == ys.size(0):
                decoded = decoded.transpose(0, 1)

            if decoded.size(-1) == self.tokenizer.vocab_size():
                logits = decoded[:, -1, :]
            else:
                logits = self.model.generator(decoded[:, -1])

            next_token = int(torch.argmax(logits, dim=1).item())
            ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long, device=self.device)], dim=0)
            if next_token == self.settings.eos_id:
                break

        return self._clean_output_ids(ys.squeeze(1).tolist())

    def _beam_decode(self, src_ids: list[int]) -> list[int]:
        beam_size = max(1, self.settings.beam_size)
        batch_first = bool(getattr(self.model, "batch_first", False))

        if batch_first:
            src = torch.tensor([src_ids], dtype=torch.long, device=self.device)
            beam: list[tuple[torch.Tensor, float]] = [
                (torch.tensor([[self.settings.sos_id]], dtype=torch.long, device=self.device), 0.0)
            ]

            for _ in range(self.settings.max_len - 1):
                new_beam: list[tuple[torch.Tensor, float]] = []
                for seq, score in beam:
                    if int(seq[0, -1].item()) == self.settings.eos_id:
                        new_beam.append((seq, score))
                        continue

                    logits = self.model(src, seq)
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

                    for i in range(beam_size):
                        next_id = int(topk_ids[0, i].item())
                        new_seq = torch.cat(
                            [seq, torch.tensor([[next_id]], dtype=torch.long, device=self.device)],
                            dim=1,
                        )
                        new_score = score + float(topk_log_probs[0, i].item())
                        new_beam.append((new_seq, new_score))

                beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]
                if all(int(seq[0, -1].item()) == self.settings.eos_id for seq, _ in beam):
                    break

            best_seq = beam[0][0].squeeze(0).tolist()
            return self._clean_output_ids(best_seq)

        src = torch.tensor(src_ids, dtype=torch.long, device=self.device).unsqueeze(1)
        src_mask = torch.zeros((src.size(0), src.size(0)), device=self.device, dtype=torch.bool)
        memory = self.model.encode(src, src_mask)

        beam = [(torch.tensor([[self.settings.sos_id]], dtype=torch.long, device=self.device), 0.0)]
        for _ in range(self.settings.max_len - 1):
            new_beam: list[tuple[torch.Tensor, float]] = []
            for seq, score in beam:
                if int(seq[-1, 0].item()) == self.settings.eos_id:
                    new_beam.append((seq, score))
                    continue

                tgt_mask = generate_square_subsequent_mask(seq.size(0), self.device)
                decoded = self.model.decode(seq, memory, tgt_mask)
                if decoded.dim() == 3 and decoded.size(0) == seq.size(0):
                    decoded = decoded.transpose(0, 1)

                if decoded.size(-1) == self.tokenizer.vocab_size():
                    logits = decoded[:, -1, :]
                else:
                    logits = self.model.generator(decoded[:, -1])

                log_probs = F.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

                for i in range(beam_size):
                    next_id = int(topk_ids[0, i].item())
                    new_seq = torch.cat(
                        [seq, torch.tensor([[next_id]], dtype=torch.long, device=self.device)],
                        dim=0,
                    )
                    new_score = score + float(topk_log_probs[0, i].item())
                    new_beam.append((new_seq, new_score))

            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]
            if all(int(seq[-1, 0].item()) == self.settings.eos_id for seq, _ in beam):
                break

        best_seq = beam[0][0].squeeze(1).tolist()
        return self._clean_output_ids(best_seq)


def load_runtime_resources(settings: Settings) -> RuntimeResources:
    if not settings.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {settings.tokenizer_path}")

    tokenizer = spm.SentencePieceProcessor()
    if not tokenizer.load(str(settings.tokenizer_path)):
        raise RuntimeError(f"Failed to load tokenizer from: {settings.tokenizer_path}")

    vocab_size = tokenizer.vocab_size()
    model = load_transformer_model(settings, vocab_size=vocab_size)
    translator = Translator(model=model, tokenizer=tokenizer, settings=settings)

    return RuntimeResources(translator=translator)
