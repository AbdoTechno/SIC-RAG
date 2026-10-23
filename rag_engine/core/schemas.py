"""Unified data contracts for document processing, chunking, and evidence retrieval."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class Document:
    """Represents a raw or parsed source document."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    """Represents a text chunk split from a document."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Evidence:
    """Represents retrieved context evidence used for generation."""
    text: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
