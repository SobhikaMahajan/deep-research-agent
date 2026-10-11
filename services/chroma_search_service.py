"""
ChromaDB-based semantic search service.
Replaces Azure AI Search with a free, local vector database.
Uses sentence-transformers for embeddings.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from the internal document store."""
    title: str
    content: str
    source: str
    page: Optional[int] = None
    score: Optional[float] = None
    document_name: Optional[str] = None
    metadata_extra: Optional[dict] = field(default_factory=dict)

    def to_snippet(self) -> str:
        """Format as a readable snippet with citations."""
        parts = [f"**{self.document_name or self.title}**"]
        parts.append(self.content)
        if self.document_name:
            parts.append(f"Document: {self.document_name}")
        if self.page is not None:
            parts.append(f"Page: {self.page}")
        if self.source:
            parts.append(f"Source: {self.source}")
        return "\n".join(parts)


class SearchServiceError(Exception):
    """Custom exception for search service errors."""
    pass


class ChromaSearchService:
    """
    Semantic search service using ChromaDB + sentence-transformers.
    Drop-in replacement for AzureSearchService with the same interface.
    """

    def __init__(self, config=None):
        """
        Initialize ChromaDB client and collection.

        Args:
            config: ChromaDBConfig instance (uses defaults if None)
        """
        from config import config as app_config

        self.config = config or app_config.chroma
        self.persist_dir = self.config.persist_dir
        self.collection_name = self.config.collection_name
        self.embedding_model = self.config.embedding_model

        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(path=self.persist_dir)

        # Use sentence-transformers for embeddings
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )

        # Get or create the collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaDB initialized: collection='{self.collection_name}', "
            f"docs={self._collection.count()}, model='{self.embedding_model}'"
        )

    def search(self, query: str, top_k: int = None) -> List[str]:
        """
        Perform semantic search against the internal document store.

        Args:
            query: The search query string
            top_k: Number of results to return (default from config)

        Returns:
            List of formatted snippet strings
        """
        top_k = top_k or self.config.default_top_k

        if self._collection.count() == 0:
            logger.warning("ChromaDB collection is empty — no documents to search")
            return ["No internal documents found. Please load documents first."]

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            parsed = self._parse_results(results)
            # Sort by score descending (lower distance = better match for cosine)
            parsed.sort(key=lambda r: r.score or float("inf"))

            return [r.to_snippet() for r in parsed]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchServiceError(f"Search failed: {e}") from e

    def _parse_results(self, results: dict) -> List[SearchResult]:
        """Parse ChromaDB query results into SearchResult objects."""
        parsed = []

        if not results or not results.get("documents"):
            return parsed

        documents = results["documents"][0]  # First query's results
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents)
        distances = results["distances"][0] if results.get("distances") else [None] * len(documents)

        for doc, meta, dist in zip(documents, metadatas, distances):
            meta = meta or {}

            # Extract metadata fields
            name = meta.get("name", meta.get("document_name", "Unknown Document"))
            # Clean name of path prefixes
            if "/" in name:
                name = name.split("/")[-1]
            if "\\" in name:
                name = name.split("\\")[-1]

            result = SearchResult(
                title=name,
                content=doc[:2000] if doc else "",  # Truncate very long content
                source=meta.get("source", meta.get("path", "")),
                page=meta.get("page_number", meta.get("page", None)),
                score=dist,
                document_name=name,
                metadata_extra={
                    k: v for k, v in meta.items()
                    if k not in ("name", "document_name", "source", "path", "page_number", "page")
                },
            )
            parsed.append(result)

        return parsed

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        Add documents to the ChromaDB collection.

        Args:
            documents: List of document text strings
            metadatas: Optional list of metadata dicts per document
            ids: Optional list of unique IDs per document
        """
        if not documents:
            logger.warning("No documents to add")
            return

        # Generate IDs if not provided
        if ids is None:
            existing_count = self._collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

        # Ensure metadatas list matches documents
        if metadatas is None:
            metadatas = [{}] * len(documents)

        # ChromaDB requires metadata values to be str, int, float, or bool
        sanitized_metadatas = []
        for meta in metadatas:
            sanitized = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    sanitized[k] = v
                elif v is None:
                    continue
                else:
                    sanitized[k] = str(v)
            sanitized_metadatas.append(sanitized)

        try:
            self._collection.add(
                documents=documents,
                metadatas=sanitized_metadatas,
                ids=ids,
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise SearchServiceError(f"Failed to add documents: {e}") from e

    @property
    def document_count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def clear_collection(self):
        """Delete and recreate the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Cleared collection '{self.collection_name}'")
