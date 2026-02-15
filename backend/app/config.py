import os
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    model_path: Path
    tokenizer_path: Path
    device: str
    max_len: int
    beam_size: int
    sos_id: int
    eos_id: int
    pad_id: int
    d_model: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float

    @classmethod
    def from_env(cls) -> "Settings":
        backend_root = Path(__file__).resolve().parents[1]
        
        def resolve_artifact_path(raw_path: str, root: Path) -> Path:
            path = Path(raw_path)
            return path if path.is_absolute() else (root / path).resolve()

        default_device = "cuda" if torch.cuda.is_available() else "cpu"

        return cls(
            model_path=resolve_artifact_path(
                os.getenv("MODEL_PATH", "models/transformer.pth"),
                backend_root,
            ),
            tokenizer_path=resolve_artifact_path(
                os.getenv("TOKENIZER_PATH", "tokenizer/tokenizer.model"),
                backend_root,
            ),
            device=os.getenv("DEVICE", default_device),
            max_len=int(os.getenv("MAX_LEN", "80")),
            beam_size=int(os.getenv("BEAM_SIZE", "1")),
            sos_id=int(os.getenv("SOS_ID", "1")),
            eos_id=int(os.getenv("EOS_ID", "2")),
            pad_id=int(os.getenv("PAD_ID", "0")),
            d_model=int(os.getenv("D_MODEL", "512")),
            nhead=int(os.getenv("NHEAD", "8")),
            num_encoder_layers=int(os.getenv("NUM_ENCODER_LAYERS", "6")),
            num_decoder_layers=int(os.getenv("NUM_DECODER_LAYERS", "6")),
            dim_feedforward=int(os.getenv("DIM_FEEDFORWARD", "2048")),
            dropout=float(os.getenv("DROPOUT", "0.1")),
        )


settings = Settings.from_env()
