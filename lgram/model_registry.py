"""
Model Registry — automatic model selection and discovery.

Manages spaCy model loading with fallback logic:
- en_core_web_sm (12 MB, baseline) — always available
- en_core_web_md (40 MB, GloVe vectors) — better similarity
- en_core_web_lg (800 MB, full vectors) — best accuracy
- sentence-transformers/all-MiniLM-L6-v2 (80 MB) — optional embeddings

Usage:
    from lgram.model_registry import get_model
    nlp, st_model, threshold = get_model("auto")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import spacy

_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "sm": {
        "name": "en_core_web_sm",
        "size": "12 MB",
        "has_vectors": False,
        "similarity_threshold": 0.65,
        "description": "Baseline — fast, small, no word vectors",
    },
    "md": {
        "name": "en_core_web_md",
        "size": "40 MB",
        "has_vectors": True,
        "similarity_threshold": 0.35,
        "description": "Balanced — GloVe 300d vectors, good for similarity",
    },
    "lg": {
        "name": "en_core_web_lg",
        "size": "800 MB",
        "has_vectors": True,
        "similarity_threshold": 0.35,
        "description": "Best accuracy — full GloVe vectors, heavy",
    },
}

_ST_MODEL_SPEC = {
    "name": "all-MiniLM-L6-v2",
    "size": "80 MB",
    "similarity_threshold": 0.35,
    "description": "Sentence transformers — best semantic similarity",
}


@dataclass
class ModelInfo:
    model_name: str
    nlp: Any
    st_model: Optional[Any]
    similarity_threshold: float
    has_vectors: bool


def get_model(
    spec: str = "auto",
    use_sentence_transformers: bool = False,
) -> ModelInfo:
    """
    Load a spaCy model by spec.

    spec options:
      - "auto": try md, fallback to sm
      - "sm": en_core_web_sm
      - "md": en_core_web_md
      - "lg": en_core_web_lg
      - full model name: "en_core_web_md"
    """
    nlp = None
    model_name = "en_core_web_sm"
    has_vectors = False
    similarity_threshold = 0.65
    st_model = None

    if spec == "auto":
        candidates = ["md", "sm"]
        for c in candidates:
            try:
                return get_model(c, use_sentence_transformers)
            except OSError:
                continue
        raise OSError("No spaCy model found. Install: python -m spacy download en_core_web_sm")

    elif spec in _MODEL_SPECS:
        info = _MODEL_SPECS[spec]
        model_name = info["name"]
        has_vectors = info["has_vectors"]
        similarity_threshold = info["similarity_threshold"]

    else:
        model_name = spec

    nlp = spacy.load(model_name)

    if use_sentence_transformers:
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer(_ST_MODEL_SPEC["name"])
            similarity_threshold = _ST_MODEL_SPEC["similarity_threshold"]
        except ImportError:
            pass

    return ModelInfo(
        model_name=model_name,
        nlp=nlp,
        st_model=st_model,
        similarity_threshold=similarity_threshold,
        has_vectors=has_vectors,
    )


def list_available() -> List[Dict[str, Any]]:
    """List available models on this system."""
    available = []
    for spec, info in _MODEL_SPECS.items():
        try:
            spacy.load(info["name"])
            status = "available"
        except OSError:
            status = "not installed"
        available.append({**info, "spec": spec, "status": status})
    return available
