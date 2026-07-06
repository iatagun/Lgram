"""
Grammar checking supplement via LLM (raw HTTP + structured output).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import List, Dict, Any

from .utils import split_sentences

_GRAMMAR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "grammar_errors",
        "schema": {
            "type": "object",
            "properties": {
                "errors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rule": {"type": "string"},
                            "message": {"type": "string"},
                            "correction": {"type": "string"},
                            "context": {"type": "string"},
                        },
                    }
                }
            },
            "required": ["errors"],
            "additionalProperties": False,
        }
    }
}

_DEEP_GRAMMAR_PROMPT = """Find grammar errors in this EFL student essay.
Return JSON only.
Focus on subject-verb agreement, missing subjects, article errors, wrong verb forms, pronoun errors, and word order.

ESSAY:
{text}"""

_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_CACHE_MAX_SIZE = 128


class DeepGrammarCheck:
    """LLM-powered deep grammar check using raw HTTP + structured output."""

    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model

    def check(self, text: str) -> List[Dict[str, Any]]:
        if len(text.split()) < 10:
            return []

        focus_text = self._select_focus_text(text)
        cache_key = self._cache_key(self._base_url, self._model, text)
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached

        prompt = _DEEP_GRAMMAR_PROMPT.format(text=focus_text)
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": _GRAMMAR_SCHEMA,
            "max_tokens": 768,
            "temperature": 0,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return []

        try:
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            raw = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""

            if not raw.strip() and reasoning.strip():
                raw = reasoning

            parsed = json.loads(raw.strip()) if raw.strip() else {}
            errors = parsed.get("errors", [])
            if not errors and isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        errors = v
                        break

            result = []
            for item in errors:
                if not isinstance(item, dict):
                    continue
                result.append({
                    "rule": str(item.get("rule", "unknown")),
                    "message": str(item.get("message", "")),
                    "correction": str(item.get("correction", "")),
                    "context": str(item.get("context", "")),
                })
            _CACHE[cache_key] = result
            if len(_CACHE) > _CACHE_MAX_SIZE:
                _CACHE.pop(next(iter(_CACHE)))
            return result
        except Exception:
            return []

    @staticmethod
    def _cache_key(base_url: str, model: str, text: str) -> str:
        return hashlib.sha256(f"{base_url}|{model}|{text}".encode("utf-8")).hexdigest()

    @staticmethod
    def _select_focus_text(text: str, max_sentences: int = 6, max_chars: int = 1200) -> str:
        text = " ".join(text.split())
        if len(text) <= max_chars:
            return text

        sentences = split_sentences(text)
        if len(sentences) <= max_sentences:
            return " ".join(sentences)[:max_chars]

        head = sentences[:2]
        tail = sentences[-2:]
        middle_start = max(2, (len(sentences) // 2) - 1)
        middle = sentences[middle_start:middle_start + 2]

        selected: List[str] = []
        for sent in head + tail + middle:
            if sent not in selected:
                selected.append(sent)

        return " ".join(selected)[:max_chars]
