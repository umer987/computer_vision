"""Application entry point with dependency preflight checks.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import importlib.util

try:
    import streamlit as st
except ImportError:
    st = None


REQUIRED_PACKAGES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "Pillow",
}


def find_missing_packages() -> list[str]:
    return [
        package_name
        for import_name, package_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(import_name) is None
    ]


def show_dependency_error(missing_packages: list[str]) -> None:
    if st is None:
        print("Campus Vision setup required.")
        print("Missing packages: " + ", ".join(missing_packages))
        print('Run: pip install -r requirements.txt')
        return

    st.set_page_config(page_title="Campus Vision Setup", page_icon="CV", layout="wide")
    st.title("Campus Vision Setup Required")
    st.error("Some Python packages required by the project are not installed.")
    st.write("Missing packages: " + ", ".join(missing_packages))
    st.code(
        """
        cd "E:\\New folder (2)\\bffp\\campus_vision"
        python -m venv .venv
        .venv\\Scripts\\activate
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        streamlit run app.py
        """,
        language="powershell",
    )
    st.info("This Python 3.14 version uses OpenCV-only face recognition.")


def main() -> None:
    missing_packages = find_missing_packages()
    if st is None:
        missing_packages = ["streamlit", *missing_packages]

    if missing_packages:
        show_dependency_error(missing_packages)
        return

    from dashboard import run_dashboard

    run_dashboard()


if __name__ == "__main__":
    main()
