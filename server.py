"""Compatibility wrapper for the modular HC01 backend."""

from app.main import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
