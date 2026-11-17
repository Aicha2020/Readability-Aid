from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import requests


# Configuration values required to connect to the local LLM service.
@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 420
    max_retries: int = 2
    temperature: float = 0.2


# Standard response object returned after an LLM call
@dataclass
class LLMResponse:
    text: str
    model: str
    raw_response: Optional[Dict[str, Any]]
    success: bool
    error_message: str = ""


# Load LLM configuration from environment variables.
def load_llm_config() -> LLMConfig:
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()

    if not base_url:
        raise ValueError("Missing environment variable: LLM_BASE_URL")
    if not api_key:
        raise ValueError("Missing environment variable: LLM_API_KEY")
    if not model:
        raise ValueError("Missing environment variable: LLM_MODEL")

    return LLMConfig(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "420")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    )


# Client used to send rewrite requests to the configured LLM
class LLMClient:
    def __init__(self, config: LLMConfig, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    # Build the chat request payload and send it to the LLM
    def rewrite_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        # Request format used by Ollama's OpenAI-compatible chat endpoint
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        return self._post_with_retries(payload)

    # Send the request, retry if needed, and return a structured response
    def _post_with_retries(self, payload: Dict[str, Any]) -> LLMResponse:
        # Ollama/OpenAI-compatible endpoint
        url = f"{self.config.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        last_error = ""

        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.info(
                    f"Calling LLM (attempt {attempt + 1}/{self.config.max_retries + 1}) "
                    f"model={self.config.model}"
                )

                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )

                response.raise_for_status()

                raw = response.json()

                # Expected response shape from an OpenAI-compatible chat endpoint
                text = raw["choices"][0]["message"]["content"].strip()

                self.logger.info("LLM call succeeded")

                return LLMResponse(
                    text=text,
                    model=self.config.model,
                    raw_response=raw,
                    success=True,
                )

            except requests.Timeout:
                last_error = f"Request timed out after {self.config.timeout_seconds} seconds"
                self.logger.warning(last_error)

            except requests.HTTPError as e:
                body = ""

                try:
                    body = e.response.text
                except Exception:
                    pass

                last_error = f"HTTP error: {e}. Response body: {body}"
                self.logger.warning(last_error)

            except Exception as e:
                last_error = f"Unexpected LLM error: {e}"
                self.logger.warning(last_error)

            if attempt < self.config.max_retries:
                time.sleep(2)

        self.logger.error(f"LLM call failed after retries: {last_error}")

        return LLMResponse(
            text="",
            model=self.config.model,
            raw_response=None,
            success=False,
            error_message=last_error,
        )

