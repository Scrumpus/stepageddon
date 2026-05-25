"""Compatibility shim — preserves `python main.py` and `uvicorn main:app`.

The real app lives in :mod:`src.main`.
"""

from src.config import settings
from src.main import app  # re-export for `uvicorn main:app`

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENV != "production",
        log_level=settings.LOG_LEVEL.lower(),
    )
