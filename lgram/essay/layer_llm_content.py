"""
Layer: LLM-Powered Content Analysis (LM Studio local-first).

Uses LM Studio's local LLM for rubric-based content evaluation.
FREE, OFFLINE, PRIVATE — no API keys, no cloud, no usage limits.

LM Studio runs a local OpenAI-compatible server at localhost:1234.
Setup: Download LM Studio (https://lmstudio.ai), load any model,
start the server. That's it.

Also supports Ollama (localhost:11434) and OpenAI (cloud) as fallbacks.
If no LLM is available, falls back to heuristic with a clear warning.

LM Studio quick setup (2 minutes):
  1. Download LM Studio: https://lmstudio.ai
  2. Load a model (Llama 3.2 3B, Qwen 2.5 7B, etc.)
  3. Go to Developer tab → Start Server (port 1234)
  4. CAEAS auto-detects and uses it — no config needed

Configuration via environment variables (optional):
  LLM_BASE_URL  — override server URL
  LLM_API_KEY   — override API key (default: "lm-studio")
  LLM_MODEL     — override model name
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from .models import Essay, LayerResult, RubricCriterion
from .utils import split_sentences

_CONTENT_PROMPT = """Rate this EFL student essay on a 0-100 scale.
Return JSON only.

ESSAY:
{text}

RUBRIC:
{criteria}"""

_CONTENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "essay_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "overall_score": {"type": "integer", "minimum": 0, "maximum": 100},
                "thesis_clarity": {"type": "number", "minimum": 0, "maximum": 1},
                "argument_development": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence_quality": {"type": "number", "minimum": 0, "maximum": 1},
                "topic_relevance": {"type": "number", "minimum": 0, "maximum": 1},
                "strengths": {"type": "array", "items": {"type": "string"}},
                "weaknesses": {"type": "array", "items": {"type": "string"}},
                "cefr_estimate": {
                    "type": "string",
                    "enum": ["A1", "A2", "B1", "B2", "C1", "C2"],
                },
            },
            "required": [
                "overall_score",
                "thesis_clarity",
                "argument_development",
                "evidence_quality",
                "topic_relevance",
                "strengths",
                "weaknesses",
                "cefr_estimate",
            ],
            "additionalProperties": False,
        },
    },
}

_CACHE: Dict[str, LayerResult] = {}
_CACHE_MAX_SIZE = 128


def _cache_put(key: str, result: LayerResult) -> None:
    _CACHE[key] = result
    if len(_CACHE) > _CACHE_MAX_SIZE:
        _CACHE.pop(next(iter(_CACHE)))


_LOCAL_ENDPOINTS = [
    ("http://localhost:1234/v1", "lm-studio", "auto"),
    ("http://localhost:11434/v1", "ollama", "auto"),
]


class LLMContentAnalyzer:
    """
    Local-first LLM content evaluation via LM Studio.

    Auto-detects running local server (LM Studio → Ollama).
    Falls back to heuristic if no local LLM found.
    All processing is local — no data leaves the machine.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = None
        self._load_error: Optional[str] = None
        self._source: str = "unknown"
        self._actual_model: str = ""

        if base_url:
            self._base_url = base_url
            self._api_key = api_key or "lm-studio"
            self._model = model or "auto"
        elif os.environ.get("LLM_BASE_URL"):
            self._base_url = os.environ["LLM_BASE_URL"]
            self._api_key = api_key or os.environ.get("LLM_API_KEY", "lm-studio")
            self._model = model or os.environ.get("LLM_MODEL", "auto")
        else:
            detected = self._detect_endpoint()
            if detected:
                self._base_url, self._api_key, self._model, self._source = detected
            else:
                self._api_key = "lm-studio"
                self._base_url = "http://localhost:1234/v1"
                self._model = "auto"

    def _detect_endpoint(self) -> Optional[tuple]:
        for url, key, model in _LOCAL_ENDPOINTS:
            try:
                import urllib.request

                req = urllib.request.Request(f"{url}/models", method="GET")
                urllib.request.urlopen(req, timeout=2)
                source = "lm-studio" if "1234" in url else "ollama"
                return (url, key, model, source)
            except Exception:
                continue
        return None

    @property
    def available(self) -> bool:
        if self._client is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key, base_url=self._base_url, timeout=60
            )
            models = self._client.models.list()
            if models.data:
                self._model = models.data[0].id
                self._actual_model = self._model
            return True
        except Exception as e:
            self._load_error = str(e)
            return False

    @property
    def client(self):
        if not self.available:
            return None
        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def source(self) -> str:
        return self._source

    def analyze(self, essay: Essay, rubric: List[RubricCriterion]) -> LayerResult:
        cache_key = self._cache_key(essay.text, rubric)
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached

        if not self.available:
            from .layer1_content import MockContentAnalyzer

            fallback = MockContentAnalyzer()
            result = fallback.analyze(essay, rubric)
            result.layer_name = "Content (heuristic — no local LLM found)"
            _cache_put(cache_key, result)
            return result

        criteria_text = self._compact_rubric(rubric)
        excerpt = self._select_focus_text(essay.text)

        prompt = _CONTENT_PROMPT.format(
            criteria=criteria_text,
            text=excerpt,
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an EFL essay evaluator. Be fair and constructive.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=_CONTENT_SCHEMA,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            msg = response.choices[0].message

            raw = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", "") or ""

            if not raw.strip() and reasoning.strip():
                raw = reasoning

            data = self._parse_json(raw)
            if not data:
                if reasoning.strip():
                    data = self._parse_json(reasoning)
                if not data and msg.content:
                    data = self._parse_json(msg.content)

            score = float(data.get("overall_score", 0))
            if score == 0:
                from .layer1_content import MockContentAnalyzer

                fallback = MockContentAnalyzer()
                result = fallback.analyze(essay, rubric)
                result.layer_name = "Content (LLM parse failed, heuristic fallback)"
                _cache_put(cache_key, result)
                return result

            score = max(0.0, min(100.0, score))
            normalized = round(score / 100.0, 3)

            strengths = data.get("strengths", [])
            weaknesses = data.get("weaknesses", [])
            evidence = (strengths + weaknesses)[:4]
            if not evidence:
                evidence = ["Content evaluation completed"]

            cefr = data.get("cefr_estimate", "")

            ci_margin = 8.0
            ci = (max(0, score - ci_margin), min(100, score + ci_margin))

            result = LayerResult(
                layer_name=f"Content (LLM: {self._source})",
                score=round(score, 1),
                normalized_score=normalized,
                raw_details={
                    "thesis_clarity": round(float(data.get("thesis_clarity", 0)), 3),
                    "argument_development": round(
                        float(data.get("argument_development", 0)), 3
                    ),
                    "evidence_quality": round(
                        float(data.get("evidence_quality", 0)), 3
                    ),
                    "topic_relevance": round(float(data.get("topic_relevance", 0)), 3),
                    "cefr_estimate": cefr,
                    "model": self._model,
                    "source": self._source,
                },
                evidence=evidence,
                confidence_interval=(round(ci[0], 1), round(ci[1], 1)),
            )
            _cache_put(cache_key, result)
            return result

        except Exception as e:
            from .layer1_content import MockContentAnalyzer

            fallback = MockContentAnalyzer()
            result = fallback.analyze(essay, rubric)
            result.layer_name = f"Content (LLM error: {str(e)[:40]})"
            _cache_put(cache_key, result)
            return result

    @staticmethod
    def _cache_key(text: str, rubric: List[RubricCriterion]) -> str:
        norm_text = " ".join(text.split())
        rubric_bits = "|".join(
            f"{c.name}:{c.weight:.3f}:{c.description[:120]}" for c in rubric
        )
        digest = hashlib.sha256(
            f"{norm_text}\n{rubric_bits}".encode("utf-8")
        ).hexdigest()
        return digest

    @staticmethod
    def _compact_rubric(rubric: List[RubricCriterion]) -> str:
        lines = []
        for c in rubric:
            desc = " ".join(c.description.split())
            if len(desc) > 120:
                desc = desc[:117].rstrip() + "..."
            lines.append(f"- {c.name} ({c.weight * 100:.0f}%): {desc}")
        return "\n".join(lines)

    @staticmethod
    def _select_focus_text(
        text: str, max_sentences: int = 6, max_chars: int = 1800
    ) -> str:
        text = " ".join(text.split())
        if len(text) <= max_chars:
            return text

        sentences = split_sentences(text)
        if len(sentences) <= max_sentences:
            excerpt = " ".join(sentences)
            return excerpt[:max_chars]

        head = sentences[:2]
        tail = sentences[-2:]
        middle_start = max(2, (len(sentences) // 2) - 1)
        middle = sentences[middle_start : middle_start + 2]

        selected: List[str] = []
        for sent in head + tail + middle:
            if sent not in selected:
                selected.append(sent)

        excerpt = " ".join(selected)
        if len(excerpt) <= max_chars:
            return excerpt

        clipped: List[str] = []
        total = 0
        for sent in selected:
            extra = len(sent) + (1 if clipped else 0)
            if clipped and total + extra > max_chars:
                break
            clipped.append(sent)
            total += extra
        return " ".join(clipped)[:max_chars]

    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        content = content.strip()
        if not content:
            return {}
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:]) if len(lines) > 1 else content
            if content.endswith("```"):
                content = content[:-3]
        content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {}
