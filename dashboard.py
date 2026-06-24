"""Professional Streamlit dashboard for campus person identification."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

from attendance import AttendanceLogger
from camera_stream import CameraConfig, CameraStream
from database import BASE_DIR, initialize_database
from face_recognition_engine import FaceRecognitionEngine
from student_management import StudentManager


st.set_page_config(
    page_title="Campus Vision Surveillance",
    page_icon="CV",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #0e1117;
            color: #f5f7fb;
        }
        [data-testid="stSidebar"] {
            background: #141922;
        }
        .metric-card {
            padding: 18px;
            border: 1px solid #2d3748;
            border-radius: 8px;
            background: #171d28;
        }
        .metric-card h3 {
            margin: 0;
            color: #a5b4fc;
            font-size: 0.9rem;
            font-weight: 600;
        }
        .metric-card p {
            margin: 6px 0 0;
            font-size: 1.7rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_services() -> tuple[StudentManager, AttendanceLogger, FaceRecognitionEngine]:
    initialize_database()
    return StudentManager(), AttendanceLogger(), FaceRecognitionEngine()


def run_dashboard() -> None:
    inject_css()

    st.title("Campus Person Identification and Surveillance System")
    st.caption("Real-time IP camera recognition, attendance automation, and analytics.")

    show_dependency_notice()
    student_manager, attendance_logger, recognition_engine = get_services()

    menu = st.sidebar.radio(
        "Navigation",
        [
            "Live Surveillance",
            "Student Registration",
            "Registered Students",
            "Attendance Records",
            "Analytics",
            "Detection History",
            "Installation Guide",
        ],
    )

    if menu == "Live Surveillance":
        live_surveillance_page(student_manager, attendance_logger, recognition_engine)
    elif menu == "Student Registration":
        student_registration_page(student_manager)
    elif menu == "Registered Students":
        registered_students_page(student_manager)
    elif menu == "Attendance Records":
        attendance_records_page(attendance_logger)
    elif menu == "Analytics":
        analytics_page(student_manager, attendance_logger)
    elif menu == "Detection History":
        detection_history_page(attendance_logger)
    else:
        installation_page()


def live_surveillance_page(
    student_manager: StudentManager,
    attendance_logger: AttendanceLogger,
    recognition_engine: FaceRecognitionEngine,
) -> None:
    st.subheader("Live Camera Feed")
    col_settings, col_feed = st.columns([0.32, 0.68])

    with col_settings:
        camera_name = st.text_input("Detection Camera", value="Main Gate Camera")
        source_type = st.selectbox("Camera Type", ["Android IP Camera", "Laptop Webcam"])
        if source_type == "Android IP Camera":
            stream_url = st.text_input(
                "Stream URL",
                value="http://192.168.1.10:8080/video",
                help="Use DroidCam or IP Webcam video URL.",
            )
            source = stream_url
        else:
            source = st.text_input("Webcam Index", value="0")

        tolerance = st.slider("Recognition Strictness", 0.45, 0.90, 0.62, 0.01)
        duplicate_minutes = st.number_input("Duplicate Block Minutes", 1, 120, 10)
        process_every = st.slider("Process Every N Frames", 1, 10, 2)
        frames_per_refresh = st.slider("Frames Per Refresh", 5, 100, 30)
        run_camera = st.toggle("Start Surveillance", value=False)

    with col_feed:
        feed_placeholder = st.empty()
        status_placeholder = st.empty()

    if not run_camera:
        st.info("Start surveillance after connecting your phone and confirming the stream URL.")
        return

    if cv2 is None:
        st.error("OpenCV is missing, so the live camera feed cannot start.")
        st.code(
            "pip install opencv-python\npip install -r requirements.txt",
            language="powershell",
        )
        return

    recognition_engine.tolerance = tolerance
    attendance_logger.duplicate_window = pd.Timedelta(minutes=int(duplicate_minutes)).to_pytimedelta()
    known_encodings, known_metadata = student_manager.get_known_face_data()

    if not known_encodings:
        st.warning("Register at least one student before starting recognition.")
        return

    stream = CameraStream(CameraConfig(source=source, name=camera_name))
    frame_count = 0

    try:
        stream.connect()
        for _ in range(frames_per_refresh):
            frame = stream.read()
            frame_count += 1

            results = []
            if frame_count % process_every == 0:
                results = recognition_engine.recognize_frame(
                    frame, known_encodings, known_metadata
                )
                for result in results:
                    attendance_logger.log_recognition(result, camera_name, frame)

            annotated = recognition_engine.annotate_frame(frame, results, fps=stream.fps)
            feed_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
            status_placeholder.success(
                f"Camera active: {camera_name} | Faces detected: {len(results)} | FPS: {stream.fps:.1f}"
            )
    except Exception as exc:
        st.error(f"Camera error: {exc}")
    finally:
        stream.release()

    if run_camera:
        st.rerun()


def student_registration_page(student_manager: StudentManager) -> None:
    st.subheader("Add Student")
    with st.form("add_student_form", clear_on_submit=True):
        name = st.text_input("Full Name")
        student_id = st.text_input("Student ID")
        department = st.text_input("Department")
        image_file = st.file_uploader("Student Photo", type=["jpg", "jpeg", "png", "webp"])
        submitted = st.form_submit_button("Register Student")

    if submitted:
        if image_file is None:
            st.error("Please upload a clear student photo.")
        else:
            try:
                student_manager.add_student(
                    name, student_id, department, image_file, image_file.name
                )
                st.success(f"Student registered: {name}")
            except Exception as exc:
                st.error(f"Registration failed: {exc}")

    st.divider()
    st.subheader("Update Student")
    students = student_manager.get_all_students()
    if not students:
        st.info("No students registered yet.")
        return

    students_by_id = {student.student_id: student for student in students}
    selected_student_id = st.selectbox(
        "Select Student",
        list(students_by_id.keys()),
        format_func=lambda item: (
            f"{students_by_id[item].name} ({students_by_id[item].student_id})"
        ),
    )
    selected = students_by_id[selected_student_id]

    with st.form("update_student_form"):
        updated_name = st.text_input("Full Name", value=selected.name)
        updated_department = st.text_input("Department", value=selected.department)
        updated_image = st.file_uploader(
            "Replace Photo Optional", type=["jpg", "jpeg", "png", "webp"]
        )
        update_submitted = st.form_submit_button("Update Student")

    if update_submitted:
        try:
            student_manager.update_student(
                selected.student_id,
                updated_name,
                updated_department,
                updated_image,
                updated_image.name if updated_image else None,
            )
            st.success("Student updated successfully.")
        except Exception as exc:
            st.error(f"Update failed: {exc}")


def registered_students_page(student_manager: StudentManager) -> None:
    st.subheader("Registered Students")
    query = st.text_input("Search Student")
    students = student_manager.search_students(query) if query else student_manager.get_all_students()

    if not students:
        st.info("No matching students found.")
        return

    for student in students:
        with st.container(border=True):
            col_image, col_info, col_actions = st.columns([0.18, 0.57, 0.25])
            with col_image:
                image_path = Path(student.image_path)
                if image_path.exists():
                    st.image(Image.open(image_path), width=120)
            with col_info:
                st.markdown(f"**{student.name}**")
                st.write(f"ID: {student.student_id}")
                st.write(f"Department: {student.department}")
            with col_actions:
                if st.button("Delete", key=f"delete_{student.student_id}"):
                    try:
                        student_manager.delete_student(student.student_id)
                        st.success("Student deleted.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")


def attendance_records_page(attendance_logger: AttendanceLogger) -> None:
    st.subheader("Attendance Records")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        start = st.date_input("Start Date", value=date.today().replace(day=1))
    with col_b:
        end = st.date_input("End Date", value=date.today())
    with col_c:
        query = st.text_input("Search")

    start_value = start.isoformat() if start else None
    end_value = end.isoformat() if end else None
    records = attendance_logger.get_attendance_records(start_value, end_value, query)

    st.dataframe(records, use_container_width=True, hide_index=True)

    if not records.empty:
        csv = records.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Attendance to CSV",
            csv,
            file_name="attendance_export.csv",
            mime="text/csv",
        )
        if st.button("Save CSV Report"):
            destination = attendance_logger.export_attendance_csv(records)
            st.success(f"Report saved to {destination}")


def analytics_page(student_manager: StudentManager, attendance_logger: AttendanceLogger) -> None:
    st.subheader("Attendance Statistics")
    students = student_manager.get_all_students()
    attendance = attendance_logger.get_attendance_records()
    departments = sorted({student.department for student in students})

    col_1, col_2, col_3 = st.columns(3)
    metric_card(col_1, "Registered Students", len(students))
    metric_card(col_2, "Attendance Entries", len(attendance))
    metric_card(col_3, "Departments", len(departments))

    department_summary = attendance_logger.get_department_summary()
    daily_summary = attendance_logger.get_daily_summary()

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        st.markdown("#### Department-wise Attendance")
        if department_summary.empty:
            st.info("No attendance data available.")
        else:
            st.bar_chart(
                department_summary,
                x="department",
                y="attendance_count",
                color="#22c55e",
            )

    with chart_col_2:
        st.markdown("#### Daily Attendance Report")
        if daily_summary.empty:
            st.info("No daily attendance data available.")
        else:
            st.line_chart(daily_summary, x="date", y="attendance_count")


def detection_history_page(attendance_logger: AttendanceLogger) -> None:
    st.subheader("Detection History")
    limit = st.slider("Records", 25, 500, 100)
    history = attendance_logger.get_detection_history(limit)
    st.dataframe(history, use_container_width=True, hide_index=True)

    if not history.empty:
        unknowns = history[history["status"] == "unknown"]
        st.write(f"Unknown detections captured: {len(unknowns)}")


def installation_page() -> None:
    st.subheader("Installation Guide")
    st.code(
        """
        cd campus_vision
        python -m venv .venv
        .venv\\Scripts\\activate
        pip install -r requirements.txt
        streamlit run app.py
        """,
        language="powershell",
    )
    st.markdown(
        """
        Android camera options:
        - DroidCam: use the video URL shown by the app.
        - IP Webcam: common URL format is `http://PHONE_IP:8080/video`.
        - If browser video works but Streamlit camera fails, try `http://PHONE_IP:8080/shot.jpg`.
        - Keep the phone and laptop on the same Wi-Fi network.
        """
    )


def show_dependency_notice() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")

    if missing:
        st.warning(
            "Missing computer-vision dependency: "
            + ", ".join(missing)
            + ". Install requirements before using registration or live recognition."
        )


def metric_card(column, label: str, value: int) -> None:
    column.markdown(
        f"""
        <div class="metric-card">
            <h3>{label}</h3>
            <p>{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    run_dashboard()
