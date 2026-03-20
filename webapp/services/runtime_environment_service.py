from __future__ import annotations

import streamlit.runtime


class RuntimeEnvironmentService:
    def ensure_streamlit_runtime(self) -> None:
        """Raise SystemExit if the app is not running inside a Streamlit session."""
        if streamlit.runtime.exists():
            return

        raise SystemExit(
            "This app must be started with Streamlit. Run: "
            "`streamlit run main.py`."
        )
