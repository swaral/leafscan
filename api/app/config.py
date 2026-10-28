import os
from dataclasses import dataclass, field
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]


def _list(name: str, default: str) -> list[str]:
    return [s.strip() for s in os.getenv(name, default).split(",") if s.strip()]


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path(os.getenv("MODEL_DIR", API_ROOT / "models"))
    db_path: Path = Path(os.getenv("DB_PATH", API_ROOT / "data" / "history.db"))
    frontend_dist: Path = Path(os.getenv("FRONTEND_DIST", API_ROOT.parent / "frontend" / "dist"))
    cors_origins: list[str] = field(default_factory=lambda: _list(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"))
    max_upload_mb: float = float(os.getenv("MAX_UPLOAD_MB", "10"))
    max_batch: int = int(os.getenv("MAX_BATCH", "20"))
    onnx_threads: int = int(os.getenv("ONNX_THREADS", "0"))


settings = Settings()
