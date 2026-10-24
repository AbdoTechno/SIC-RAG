from typing import Dict, Type
from core.interfaces import (
    BaseFileParser,
    BaseDocumentStore,
    BaseTextNormalizer,
    BaseChunker,
    BaseEmbedder,
    BaseReranker,
)

class ComponentRegistry:
    def __init__(self):
        self._parsers: Dict[str, Type[BaseFileParser]] = {}
        self._doc_stores: Dict[str, Type[BaseDocumentStore]] = {}
        self._normalizers: Dict[str, Type[BaseTextNormalizer]] = {}
        self._chunkers: Dict[str, Type[BaseChunker]] = {}
        self._embedders: Dict[str, Type[BaseEmbedder]] = {}
        self._rerankers: Dict[str, Type[BaseReranker]] = {}

    def register_parser(self, extension: str):
        ext = extension.lower().strip()
        if not ext.startswith("."):
            ext = f".{ext}"

        def decorator(cls: Type[BaseFileParser]):
            if not issubclass(cls, BaseFileParser):
                raise TypeError(f"{cls.__name__} must inherit from BaseFileParser")
            self._parsers[ext] = cls
            return cls
        return decorator

    def register_document_store(self, name: str):
        key = name.lower().strip()
        def decorator(cls: Type[BaseDocumentStore]):
            if not issubclass(cls, BaseDocumentStore):
                raise TypeError(f"{cls.__name__} must inherit from BaseDocumentStore")
            self._doc_stores[key] = cls
            return cls
        return decorator

    def register_normalizer(self, name: str):
        key = name.lower().strip()
        def decorator(cls: Type[BaseTextNormalizer]):
            if not issubclass(cls, BaseTextNormalizer):
                raise TypeError(f"{cls.__name__} must inherit from BaseTextNormalizer")
            self._normalizers[key] = cls
            return cls
        return decorator

    def register_chunker(self, name: str):
        key = name.lower().strip()
        def decorator(cls: Type[BaseChunker]):
            if not issubclass(cls, BaseChunker):
                raise TypeError(f"{cls.__name__} must inherit from BaseChunker")
            self._chunkers[key] = cls
            return cls
        return decorator

    def register_reranker(self, name: str):
        key = name.lower().strip()
        def decorator(cls: Type[BaseReranker]):
            if not issubclass(cls, BaseReranker):
                raise TypeError(f"{cls.__name__} must inherit from BaseReranker")
            self._rerankers[key] = cls
            return cls
        return decorator

    def get_parser(self, extension: str) -> Type[BaseFileParser]:
        ext = extension.lower().strip()
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext not in self._parsers:
            raise KeyError(f"Unsupported file format '{ext}'. Registered: {list(self._parsers.keys())}")
        return self._parsers[ext]

    def get_document_store(self, name: str) -> Type[BaseDocumentStore]:
        key = name.lower().strip()
        if key not in self._doc_stores:
            raise KeyError(f"Unknown document store '{key}'. Registered: {list(self._doc_stores.keys())}")
        return self._doc_stores[key]

    def get_normalizer(self, name: str) -> Type[BaseTextNormalizer]:
        key = name.lower().strip()
        if key not in self._normalizers:
            raise KeyError(f"Unknown normalizer '{key}'. Registered: {list(self._normalizers.keys())}")
        return self._normalizers[key]

    def get_chunker(self, name: str) -> Type[BaseChunker]:
        key = name.lower().strip()
        if key not in self._chunkers:
            raise KeyError(f"Unknown chunker '{key}'. Registered: {list(self._chunkers.keys())}")
        return self._chunkers[key]

    def get_reranker(self, name: str) -> Type[BaseReranker]:
        key = name.lower().strip()
        if key not in self._rerankers:
            raise KeyError(f"Unknown reranker '{key}'. Registered: {list(self._rerankers.keys())}")
        return self._rerankers[key]

registry = ComponentRegistry()