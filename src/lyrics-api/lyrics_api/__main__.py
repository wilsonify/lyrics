"""
Run FastAPI behind an ASGI server
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app="lyrics_api.controller:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload_dirs=True
    )
