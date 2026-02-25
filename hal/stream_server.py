"""
hal/stream_server.py
Lightweight push-based MJPEG HTTP stream server for remote monitoring.

Serves a multipart/x-mixed-replace (MJPEG) stream at:
    http://<pi-ip>:<port>/stream

Also serves a minimal HTML wrapper at:
    http://<pi-ip>:<port>/

Usage
-----
    streamer = MJPEGStreamer(port=8080)
    streamer.start()

    # Each control tick:
    streamer.push_frame(bgr_numpy_array)

    streamer.stop()

Multiple browser clients are supported simultaneously via ThreadingMixIn.
Push-side encoding is done once per frame regardless of viewer count.
No external dependencies beyond OpenCV (already required by the project).
"""
from __future__ import annotations

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Shared state (module-level; one streamer per process) ────────
_state_lock  = threading.Lock()
_jpeg_bytes: Optional[bytes] = None
_frame_event  = threading.Event()   # signalled each time a new frame arrives

_INDEX_HTML = b"""\
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Robot Camera</title>
    <style>
      body { background:#111; margin:0; display:flex; justify-content:center;
             align-items:center; height:100vh; flex-direction:column; }
      img  { max-width:100%; border:2px solid #444; display:block; }
      #info{ color:#888; font-family:monospace; font-size:12px; margin-top:6px; }
    </style>
  </head>
  <body>
    <img id="cam" src="/frame?t=0" alt="camera">
    <div id="info">loading...</div>
    <script>
      var frames=0, last=Date.now();
      var img = document.getElementById('cam');
      img.onload = function() {
        frames++;
        var now = Date.now();
        if (now - last >= 1000) {
          document.getElementById('info').textContent =
            Math.round(frames*1000/(now-last)) + ' fps';
          frames = 0; last = now;
        }
        // Schedule next frame immediately after this one loads
        img.src = '/frame?t=' + Date.now();
      };
      img.onerror = function() {
        document.getElementById('info').textContent = 'error - retrying';
        setTimeout(function(){ img.src = '/frame?t=' + Date.now(); }, 500);
      };
    </script>
  </body>
</html>
"""


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTPServer with one daemon thread per request, capped at 10 threads.
    Prevents thread exhaustion from stale polling connections."""
    daemon_threads = True

    def process_request(self, request, client_address):
        import threading
        if threading.active_count() > 10:
            # Too many threads - handle inline rather than spawn a new one
            self.finish_request(request, client_address)
            self.shutdown_request(request)
        else:
            super().process_request(request, client_address)


class _StreamHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — / serves the HTML page, /stream is MJPEG."""

    def log_message(self, fmt, *args) -> None:  # silence per-request access log
        pass

    def do_GET(self) -> None:
        path = self.path.split("?")[0]   # strip cache-bust query string
        if path == "/":
            self._serve_index()
        elif path == "/frame":
            self._serve_frame()
        else:
            self.send_error(404)

    def _serve_index(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(_INDEX_HTML)))
        self.end_headers()
        self.wfile.write(_INDEX_HTML)

    def _serve_frame(self) -> None:
        """Serve the latest frame as a single JPEG (polled by JS)."""
        with _state_lock:
            data = _jpeg_bytes
        if data is None:
            self.send_error(503, "No frame yet")
            return
        self.send_response(200)
        self.send_header("Content-Type",   "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control",  "no-cache, no-store, must-revalidate")
        self.send_header("Pragma",         "no-cache")
        self.send_header("Expires",        "0")
        self.end_headers()
        self.wfile.write(data)


class MJPEGStreamer:
    """
    Push-based MJPEG streamer.

    Parameters
    ----------
    port : int
        TCP port to listen on (default 8080).
    jpeg_quality : int
        JPEG encoding quality 1-100 (default 70 – good balance on RPi).
    """

    def __init__(self, port: int = 8080, jpeg_quality: int = 70) -> None:
        self._port    = port
        self._quality = jpeg_quality
        self._server: Optional[_ThreadingHTTPServer] = None
        self._thread:  Optional[threading.Thread]    = None

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background HTTP server thread."""
        if self._server is not None:
            return  # already running
        try:
            self._server = _ThreadingHTTPServer(("", self._port), _StreamHandler)
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="mjpeg-server",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "[Stream] MJPEG server  http://0.0.0.0:%d/  (stream: /stream)",
                self._port,
            )
        except OSError as exc:
            logger.error("[Stream] Failed to start MJPEG server on port %d: %s",
                         self._port, exc)
            self._server = None

    def stop(self) -> None:
        """Shut down the HTTP server gracefully."""
        if self._server:
            self._server.shutdown()
            self._server = None
            logger.info("[Stream] MJPEG server stopped")

    # ── Frame push ────────────────────────────────────────────────

    def push_frame(self, bgr: np.ndarray) -> None:
        """
        JPEG-encode *bgr* and notify all streaming clients.

        This is called from the main control loop and must be fast.
        Encoding is O(pixels) on CPU — at 640×480 with quality=70 it
        takes ~3 ms on RPi 5, well within a 100 ms tick budget.
        """
        if self._server is None:
            return
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        ok, buf = cv2.imencode(".jpg", bgr, encode_params)
        if not ok:
            return
        global _jpeg_bytes
        with _state_lock:
            _jpeg_bytes = buf.tobytes()
        _frame_event.set()
