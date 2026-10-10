"""Maintenance helpers; invoke only from a protected admin workflow."""


def clear_streamlit_resource_cache(streamlit_module) -> None:
    """Clear cached shared resources; the next app run reloads model and index."""
    streamlit_module.cache_resource.clear()
