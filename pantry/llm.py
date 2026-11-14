# LLM wrappers: the abstract LLM, OpenNextLLM (OpenAI-compatible endpoint) and the deterministic FakeLLM.
# chat_json validates every response with a pydantic schema and retries once with the error before raising.
from __future__ import annotations

import hashlib
import json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from openai import OpenAI as SdkOpenAI
from pydantic import BaseModel, ValidationError

from pantry.config import Settings, get_settings

DEFAULT_EMBED_MODEL: str = "text-embedding-3-small"
KNOWN_EMBED_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-MiniLM-L6-v2": 384,
}
MAX_ATTEMPTS: int = 2  # first try plus one retry with the validation error

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
_TOKEN = re.compile(r"[a-z0-9]+")


class LLMError(Exception):
    """Base class for LLM wrapper errors."""


class LLMConfigError(LLMError):
    """Raised when the real LLM cannot be built because configuration is missing."""


class LLMOutputError(LLMError):
    """Raised when the LLM output is still invalid after the retry; callers switch to their fallback."""

    def __init__(self, schema_name: str, attempts: int, last_error: str, last_output: str) -> None:
        super().__init__(
            f"LLM output failed {schema_name} validation after {attempts} attempts: {last_error}"
        )
        self.schema_name = schema_name
        self.attempts = attempts
        self.last_error = last_error
        self.last_output = last_output


def embedding_dimension(model: str) -> int:
    """Return the known vector dimension for an embedding model (1536 when unknown)."""
    return KNOWN_EMBED_DIMENSIONS.get(model, 1536)


def prompt_key(system: str, user: str) -> str:
    """Stable key for a (system, user) prompt pair, used to register FakeLLM responses."""
    return hashlib.sha256(f"{system}\n\x00\n{user}".encode("utf-8")).hexdigest()[:16]


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a surrounding Markdown code fence."""
    match = _FENCE.match(text)
    payload = match.group(1) if match else text
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data


def _schema_instruction(schema: type[BaseModel]) -> str:
    """System prompt suffix describing the required JSON output."""
    return (
        "\n\nRespond with a single JSON object only, no prose, matching this JSON schema:\n"
        + json.dumps(schema.model_json_schema(), separators=(",", ":"))
    )


class LLM(ABC):
    """Abstract LLM: structured JSON chat and text embeddings."""

    def chat_json(self, system: str, user: str, schema: type[BaseModel]) -> dict[str, Any]:
        """Return the model's JSON answer validated against schema (as a JSON-compatible dict).

        On invalid JSON or a validation error, retry once with the error message appended; if it still
        fails, raise LLMOutputError.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system + _schema_instruction(schema)},
            {"role": "user", "content": user},
        ]
        last_error = ""
        last_output = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            last_output = self._complete(
                messages, system=system, user=user, schema_name=schema.__name__, attempt=attempt
            )
            try:
                return schema.model_validate(_parse_json_object(last_output)).model_dump(mode="json")
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
            messages = [
                *messages,
                {"role": "assistant", "content": last_output},
                {
                    "role": "user",
                    "content": (
                        f"Your previous output was invalid: {last_error}\n"
                        "Return a corrected JSON object that matches the schema exactly."
                    ),
                },
            ]
        raise LLMOutputError(schema.__name__, MAX_ATTEMPTS, last_error, last_output)

    @abstractmethod
    def _complete(
        self, messages: list[dict[str, str]], *, system: str, user: str, schema_name: str, attempt: int
    ) -> str:
        """Return the raw text of one chat completion for messages (attempt starts at 1)."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in input order."""


class OpenNextLLM(LLM):
    """Real LLM on the OpenNext OpenAI-compatible endpoint: JSON mode, temperature 0."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._settings = settings or get_settings()
        self.model = self._settings.llm_model
        self.embed_model = self._settings.embed_model or DEFAULT_EMBED_MODEL
        if client is None:
            missing = [
                name
                for name, value in (
                    ("LLM_BASE_URL", self._settings.llm_base_url),
                    ("LLM_API_KEY", self._settings.llm_api_key),
                    ("LLM_MODEL", self._settings.llm_model),
                )
                if not value
            ]
            if missing:
                raise LLMConfigError(f"OpenNextLLM needs {', '.join(missing)}; use FakeLLM instead")
            client = SdkOpenAI(
                base_url=self._settings.llm_base_url,
                api_key=self._settings.llm_api_key,
                timeout=timeout,
            )
        self._client = client

    def _complete(
        self, messages: list[dict[str, str]], *, system: str, user: str, schema_name: str, attempt: int
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.embed_model, input=list(texts))
        data = sorted(response.data, key=lambda d: d.index)
        return [list(d.embedding) for d in data]


class FakeLLM(LLM):
    """Deterministic offline LLM: canned JSON responses per prompt key and hash-based embeddings.

    Responses are looked up by prompt_key(system, user) first, then by the schema class name. A key may
    hold a sequence of responses consumed one per call (the last one repeats), which lets tests register
    an invalid answer followed by a valid one to exercise the retry.
    """

    def __init__(self, embed_dim: int | None = None, embed_model: str = DEFAULT_EMBED_MODEL) -> None:
        self.embed_model = embed_model
        self.embed_dim = embed_dim or embedding_dimension(embed_model)
        self._responses: dict[str, list[Any]] = {}
        self._cursor: dict[str, int] = {}
        self.calls: list[dict[str, Any]] = []

    def register(self, key: str, *responses: Mapping[str, Any] | str) -> None:
        """Register canned responses for a prompt key or a schema class name (dicts or raw strings)."""
        if not responses:
            raise ValueError("register needs at least one response")
        self._responses[key] = list(responses)
        self._cursor[key] = 0

    def register_prompt(self, system: str, user: str, *responses: Mapping[str, Any] | str) -> None:
        """Register canned responses for an exact (system, user) prompt pair."""
        self.register(prompt_key(system, user), *responses)

    def _complete(
        self, messages: list[dict[str, str]], *, system: str, user: str, schema_name: str, attempt: int
    ) -> str:
        key = prompt_key(system, user)
        if key not in self._responses:
            key = schema_name
        if key not in self._responses:
            raise LLMError(
                f"FakeLLM has no canned response for prompt key {prompt_key(system, user)} or schema {schema_name}"
            )
        queue = self._responses[key]
        index = min(self._cursor[key], len(queue) - 1)
        self._cursor[key] += 1
        self.calls.append({"key": key, "attempt": attempt, "system": system, "user": user})
        response = queue[index]
        return response if isinstance(response, str) else json.dumps(response)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._hash_embedding(text) for text in texts]

    def _hash_embedding(self, text: str) -> list[float]:
        """Feature-hash lowercase tokens into embed_dim signed buckets and L2-normalise.

        Texts sharing tokens get high cosine similarity, which keeps retrieval tests meaningful.
        """
        vec = [0.0] * self.embed_dim
        tokens = _TOKEN.findall(text.lower()) or [text]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.embed_dim
            vec[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:4], "big") % self.embed_dim] = 1.0
            return vec
        return [v / norm for v in vec]
