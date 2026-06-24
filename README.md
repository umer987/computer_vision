# Campus Person Identification and Surveillance System

A modular computer vision project for real-time university campus person identification, attendance logging, and Streamlit surveillance analytics.

This version is built for Python 3.14.3. It uses OpenCV for face detection and recognition so it does not require `face_recognition` or `dlib`.

## Features

- Student registration with photo upload, metadata, and facial encodings
- SQLite database for students, attendance, and detection history
- Android phone IP camera support through DroidCam or IP Webcam
- OpenCV Haar cascade multi-face detection
- OpenCV face-template matching for registered student recognition
- Green bounding boxes for registered students and red boxes for unknown people
- Attendance logging with configurable duplicate prevention
- Unknown person screenshot capture
- Live FPS display and confidence scores
- Dark Streamlit dashboard with student search, reports, and charts
- CSV attendance export

## Folder Structure

```text
campus_vision/
|-- app.py
|-- database.py
|-- face_recognition_engine.py
|-- attendance.py
|-- camera_stream.py
|-- student_management.py
|-- dashboard.py
|-- requirements.txt
|-- database/
|   |-- campus.db
|-- student_images/
|-- attendance_reports/
|-- unknown_captures/
```

The database file is created automatically on first run.

## Database Schema

### Students

| Column | Type | Description |
| --- | --- | --- |
| student_id | TEXT PRIMARY KEY | University student ID |
| name | TEXT | Full name |
| department | TEXT | Department name |
| image_path | TEXT | Saved registration image |
| encoding | BLOB | Pickled OpenCV face template |
| created_at | TEXT | Creation timestamp |
| updated_at | TEXT | Last update timestamp |

### Attendance

| Column | Type | Description |
| --- | --- | --- |
| attendance_id | INTEGER PRIMARY KEY | Attendance row ID |
| student_id | TEXT | Linked student ID |
| name | TEXT | Student name snapshot |
| department | TEXT | Department snapshot |
| date | TEXT | Detection date |
| time | TEXT | Detection time |
| camera_source | TEXT | Camera name/source |
| created_at | TEXT | Insert timestamp |

### Detection History

Stores every recognition event, including unknown detections, confidence scores, camera source, and screenshot paths.

## Installation for Python 3.14.3

```powershell
cd "E:\New folder (2)\bffp\campus_vision"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If your terminal uses another Python version, use the full Python 3.14.3 path or launcher command for the virtual environment.

## Android IP Camera Setup

1. Install DroidCam or IP Webcam on the Android phone.
2. Connect the phone and laptop to the same Wi-Fi network.
3. Start the camera server in the mobile app.
4. Copy the video stream URL into the dashboard.
5. Common IP Webcam URL format:

```text
http://PHONE_IP:8080/video
```

## Running the System

1. Open the Streamlit dashboard.
2. Register students from the Student Registration page.
3. Go to Live Surveillance.
4. Enter the camera source and camera name.
5. Start surveillance.

Recognized faces show:

```text
Name: Umer Shakir
ID: SE-2023-001
Department: Software Engineering
Confidence: 91.5%
```

Unknown faces show:

```text
Unknown Person
```

## Accuracy Notes

This Python 3.14.3 build uses a lightweight OpenCV template matcher. It is easier to install for university demos, but it is less robust than deep-learning face embeddings. For best results:

- Register clear frontal photos.
- Use good lighting.
- Keep camera angle close to the registration image angle.
- Register one person per photo.
- Increase Recognition Strictness if false matches occur.
- Decrease Recognition Strictness if registered students are often marked unknown.

## Presentation Tips

- Register 3 to 5 students with clear frontal images.
- Use the laptop webcam first for a quick dry run.
- Then switch to the Android IP camera stream for the CCTV demo.
- Show the attendance table, department chart, daily report, and unknown capture history.
