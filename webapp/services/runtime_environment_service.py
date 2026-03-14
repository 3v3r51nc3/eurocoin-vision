from __future__ import annotations

import streamlit.runtime


class RuntimeEnvironmentService:
    def ensure_streamlit_runtime(self) -> None:
        if streamlit.runtime.exists():
            return

        raise SystemExit(
            "This app must be started with Streamlit. Run: "
            "`streamlit run main.py`."
        )
