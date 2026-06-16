"""
Web Dashboard — hiển thị 3 nguồn SoC song song.

Module này khởi động một Flask web server để hiển thị:
  - 3 battery icons (SoC #1 BMS, #2 Coulomb Counter, #3 CNN1D model)
  - Range estimation, SoH, Wh/km baseline
  - Real-time updates via AJAX polling

Kết nối với main.py qua /api/state endpoint (localhost:5000/api/state).

Cấu trúc:
  - src/display/web.py (Flask app)
  - src/display/templates/index.html (dashboard HTML + CSS + JS)
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"


def create_app(api_endpoint: str = "http://localhost:5000/api/state") -> object:
    """
    Tạo Flask app để display dashboard.

    Args:
        api_endpoint: URL endpoint của /api/state từ main.py.

    Returns:
        Flask app instance.
    """
    try:
        from flask import Flask, render_template
    except ImportError:
        logger.error("Flask not installed. Install with: pip install flask")
        raise

    app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
    app.config["API_ENDPOINT"] = api_endpoint

    @app.route("/")
    def index():
        """Hiển thị dashboard."""
        return render_template("index.html", api_endpoint=api_endpoint)

    @app.route("/health")
    def health():
        """Health check endpoint."""
        return {"status": "ok"}

    logger.info(f"Flask app created (API endpoint: {api_endpoint})")
    return app


def run_server(
    app: object = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    api_endpoint: str = "http://localhost:5000/api/state",
    debug: bool = False,
) -> None:
    """
    Khởi động Flask web server.

    Args:
        app: Flask app instance. Nếu None, tạo mới.
        host: Host to bind to (mặc định 0.0.0.0).
        port: Port to bind to (mặc định 8080 để tránh xung đột với main.py:5000).
        api_endpoint: URL endpoint của /api/state từ main.py.
        debug: Debug mode (không dùng trong production).
    """
    if app is None:
        app = create_app(api_endpoint=api_endpoint)

    try:
        logger.info(f"Starting web server at {host}:{port} (debug={debug})")
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    except Exception as e:
        logger.error(f"Web server error: {e}")
        raise


if __name__ == "__main__":
    # Standalone mode — run web server pointing to local main.py API
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    run_server(
        host="0.0.0.0",
        port=8080,
        api_endpoint="http://localhost:5000/api/state",
        debug=False,
    )
