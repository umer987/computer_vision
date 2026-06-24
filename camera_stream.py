"""Camera stream abstraction for webcams and Android IP camera apps."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class CameraConfig:
    source: str | int
    name: str = "Campus Camera 1"
    width: int = 1280
    height: int = 720


class CameraStream:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.capture: cv2.VideoCapture | None = None
        self.snapshot_url: str | None = None
        self.previous_frame_time = time.perf_counter()
        self.fps = 0.0

    def connect(self) -> None:
        if cv2 is None:
            raise ImportError(
                "OpenCV is not installed. Run `pip install opencv-python` or "
                "`pip install -r requirements.txt` before starting the camera."
            )
        source = self._normalize_source(self.config.source)
        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            self.snapshot_url = self._snapshot_url_from_source(source)
            if self.snapshot_url is None:
                raise ConnectionError(f"Unable to open camera stream: {source}")

            # Validate the fallback immediately so the dashboard reports a clear error.
            self._read_snapshot_frame()

    def read(self):
        if self.capture is None and self.snapshot_url is None:
            self.connect()

        if self.snapshot_url:
            frame = self._read_snapshot_frame()
        else:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                fallback_url = self._snapshot_url_from_source(self.config.source)
                if fallback_url:
                    self.snapshot_url = fallback_url
                    self.release()
                    frame = self._read_snapshot_frame()
                else:
                    raise ConnectionError("Camera frame could not be read. Check the stream URL.")

        now = time.perf_counter()
        elapsed = now - self.previous_frame_time
        self.previous_frame_time = now
        self.fps = 1.0 / elapsed if elapsed > 0 else 0.0
        return frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _read_snapshot_frame(self):
        if self.snapshot_url is None:
            raise ConnectionError("Snapshot URL is not configured.")

        try:
            request = urllib.request.Request(
                self.snapshot_url,
                headers={"User-Agent": "CampusVision/1.0"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                image_bytes = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ConnectionError(
                f"Unable to read snapshot frame from {self.snapshot_url}. "
                "For IP Webcam, try http://PHONE_IP:8080/shot.jpg."
            ) from exc

        frame_array = cv2.imdecode(np.frombuffer(image_bytes, dtype="uint8"), cv2.IMREAD_COLOR)
        if frame_array is None:
            raise ConnectionError(f"Snapshot URL did not return a valid image: {self.snapshot_url}")
        return frame_array

    @staticmethod
    def _normalize_source(source: str | int) -> str | int:
        if isinstance(source, int):
            return source
        source = source.strip()
        if source.isdigit():
            return int(source)
        return source

    @staticmethod
    def _snapshot_url_from_source(source: str | int) -> str | None:
        if not isinstance(source, str):
            return None

        source = source.strip()
        if not source.startswith(("http://", "https://")):
            return None

        parts = urlsplit(source)
        path = parts.path.rstrip("/")
        if path.endswith("/shot.jpg"):
            return source

        # IP Webcam exposes single JPEG frames here. This fallback works when
        # browser video is visible but OpenCV cannot decode the MJPEG stream.
        return urlunsplit((parts.scheme, parts.netloc, "/shot.jpg", "", ""))
