"""
Grammar checking supplement via LLM (raw HTTP + structured output).
"""
from __future__ import annotations

import json
import urllib.request
from typing import List, Dict, Any

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
        }
    }
}

_DEEP_GRAMMAR_PROMPT = """Find ALL grammar errors in this EFL student essay. Focus on subject-verb agreement, missing subjects, article errors, wrong verb forms, and pronoun errors.

ESSAY:
{text}"""


class DeepGrammarCheck:
    """LLM-powered deep grammar check using raw HTTP + structured output."""

    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model

    def check(self, text: str) -> List[Dict[str, Any]]:
        if len(text.split()) < 10:
            return []

        prompt = _DEEP_GRAMMAR_PROMPT.format(text=text[:3000])
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": _GRAMMAR_SCHEMA,
            "max_tokens": 1024,
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
            return result
        except Exception:
            return []
