"""Face detection and recognition engine using OpenCV only.

This version is designed for Python 3.14 compatibility. It avoids the
`face_recognition`/`dlib` dependency and stores a normalized grayscale face
template for each registered student.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # Allows Streamlit to render the installation guide.
    cv2 = None


FACE_SIZE = (120, 120)


@dataclass(frozen=True)
class RecognitionResult:
    top: int
    right: int
    bottom: int
    left: int
    status: str
    name: str
    student_id: str | None
    department: str | None
    confidence: float


class FaceRecognitionEngine:
    def __init__(self, tolerance: float = 0.62, detection_model: str = "haar") -> None:
        self.tolerance = tolerance
        self.detection_model = detection_model
        self._cascade = None

    @staticmethod
    def require_dependency() -> None:
        if cv2 is None:
            raise ImportError(
                "OpenCV is not installed. Run `pip install opencv-python` or "
                "`pip install -r requirements.txt` inside the campus_vision environment."
            )

    def extract_single_face_encoding(self, image_path: str | Path) -> np.ndarray:
        """Extract one normalized face template from a registration image."""
        self.require_dependency()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Could not read the uploaded image.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray)

        if len(faces) == 0:
            raise ValueError("No face was detected in the uploaded image.")
        if len(faces) > 1:
            raise ValueError("Multiple faces were detected. Upload one clear student photo.")

        x, y, width, height = faces[0]
        face_crop = gray[y : y + height, x : x + width]
        return self._build_template(face_crop)

    def recognize_frame(
        self,
        frame: np.ndarray,
        known_encodings: list[np.ndarray],
        known_metadata: list[dict[str, Any]],
        resize_scale: float = 0.5,
    ) -> list[RecognitionResult]:
        self.require_dependency()
        if frame is None or frame.size == 0:
            return []

        small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray)
        scale = 1.0 / resize_scale

        results: list[RecognitionResult] = []
        for x, y, width, height in faces:
            face_crop = gray[y : y + height, x : x + width]
            face_template = self._build_template(face_crop)
            match = self._match_face(face_template, known_encodings, known_metadata)

            left = int(x * scale)
            top = int(y * scale)
            right = int((x + width) * scale)
            bottom = int((y + height) * scale)
            results.append(
                RecognitionResult(
                    top=top,
                    right=right,
                    bottom=bottom,
                    left=left,
                    **match,
                )
            )
        return results

    def annotate_frame(
        self, frame: np.ndarray, results: list[RecognitionResult], fps: float | None = None
    ) -> np.ndarray:
        self.require_dependency()
        annotated = frame.copy()

        for result in results:
            recognized = result.status == "recognized"
            color = (0, 200, 0) if recognized else (0, 0, 255)
            cv2.rectangle(
                annotated,
                (result.left, result.top),
                (result.right, result.bottom),
                color,
                2,
            )

            lines = (
                [
                    f"Name: {result.name}",
                    f"ID: {result.student_id}",
                    f"Department: {result.department}",
                    f"Confidence: {result.confidence:.1f}%",
                ]
                if recognized
                else ["Unknown Person"]
            )
            self._draw_label(annotated, result.left, result.bottom, lines, color)

        if fps is not None:
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _detect_faces(self, gray_image: np.ndarray) -> list[tuple[int, int, int, int]]:
        cascade = self._get_cascade()
        faces = cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        return [tuple(map(int, face)) for face in faces]

    def _get_cascade(self):
        self.require_dependency()
        if self._cascade is None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(str(cascade_path))
            if self._cascade.empty():
                raise RuntimeError("OpenCV Haar cascade file could not be loaded.")
        return self._cascade

    def _build_template(self, face_crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_crop, FACE_SIZE)
        equalized = cv2.equalizeHist(resized)
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        template = blurred.astype("float32").flatten()
        norm = float(np.linalg.norm(template))
        if norm == 0:
            raise ValueError("Face template is empty and cannot be encoded.")
        return template / norm

    def _match_face(
        self,
        face_template: np.ndarray,
        known_encodings: list[np.ndarray],
        known_metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not known_encodings:
            return self._unknown_result(0.0)

        similarities = np.array(
            [float(np.dot(face_template, known_template)) for known_template in known_encodings]
        )
        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])
        confidence = max(0.0, min(100.0, best_similarity * 100.0))

        if best_similarity >= self.tolerance:
            student = known_metadata[best_index]
            return {
                "status": "recognized",
                "name": student["name"],
                "student_id": student["student_id"],
                "department": student["department"],
                "confidence": confidence,
            }
        return self._unknown_result(confidence)

    @staticmethod
    def _unknown_result(confidence: float) -> dict[str, Any]:
        return {
            "status": "unknown",
            "name": "Unknown Person",
            "student_id": None,
            "department": None,
            "confidence": confidence,
        }

    @staticmethod
    def _draw_label(
        frame: np.ndarray, left: int, bottom: int, lines: list[str], color: tuple[int, int, int]
    ) -> None:
        line_height = 22
        box_height = line_height * len(lines) + 10
        y1 = min(frame.shape[0] - 5, bottom + box_height)
        y0 = max(0, y1 - box_height)
        x0 = max(0, left)
        x1 = min(frame.shape[1] - 5, x0 + 390)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        for index, line in enumerate(lines):
            y = y0 + 20 + index * line_height
            cv2.putText(
                frame,
                line,
                (x0 + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
