from __future__ import annotations

import time

import requests


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 180,
                 temperature: float = 0.0, num_ctx: int = 8192):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._cache: tuple[float, bool, str] | None = None

    def status(self) -> tuple[bool, str]:
        """(available, human-readable reason). Cached for 15 s."""
        if self._cache and time.time() - self._cache[0] < 15:
            return self._cache[1], self._cache[2]
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            ok = any(n == self.model or n.split(":")[0] == self.model or n == f"{self.model}:latest"
                     for n in names)
            reason = f"Ollama model '{self.model}' ready" if ok else \
                f"Ollama is running but '{self.model}' is not pulled (run: ollama pull {self.model})"
        except Exception:  # noqa: BLE001
            ok, reason = False, f"Ollama not reachable at {self.base_url}"
        self._cache = (time.time(), ok, reason)
        return ok, reason

    def available(self) -> bool:
        return self.status()[0]

    def chat_json(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]
