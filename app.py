"""Streamlit application startup entry point."""

from platforms.streamlit.app import main


# Streamlit executes app scripts in its own synthetic ``__main__`` module.
# Start the UI at module execution time instead of using a Python main guard.
main()
