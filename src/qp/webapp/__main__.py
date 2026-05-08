"""Entry point: ``uv run python -m qp.webapp``.

Launches the FastAPI app on 127.0.0.1:8765 by default. Override host /
port with ``QP_WEBAPP_HOST`` / ``QP_WEBAPP_PORT`` env vars.
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("QP_WEBAPP_HOST", "127.0.0.1")
    port = int(os.environ.get("QP_WEBAPP_PORT", "8765"))
    uvicorn.run("qp.webapp.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
