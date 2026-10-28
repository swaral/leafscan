"""Dev entry point:  python run.py  (serves on http://127.0.0.1:8000, docs at /docs)."""
from pathlib import Path

import uvicorn

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True,
                app_dir=str(HERE), reload_dirs=[str(HERE / "app")])
