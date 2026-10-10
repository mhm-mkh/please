"""Small OpenAI-compatible HTTP client, configured entirely by environment variables."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMClient:
    def __init__(self) -> None:
        self.api_key = self._setting("LLM_API_KEY", "").strip()
        self.base_url = self._setting("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = self._setting("LLM_MODEL", "").strip()

    @staticmethod
    def _setting(name: str, default: str) -> str:
        value = os.getenv(name)
        if value:
            return value
        try:
            import streamlit as st
            return str(st.secrets.get(name, default))
        except Exception:
            return default

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)

    def generate(self, prompt: dict) -> str | None:
        if not self.configured:
            return None
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            # Temporary diagnostic allowance: the provider's own SDK example
            # uses a much longer default timeout than this raw HTTP client did.
            with urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
