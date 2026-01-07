"""
Service d'embeddings via Google Gemini
Utilise gemini-embedding-001 (768 dimensions, gratuit)
"""

import logging
from typing import List
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service pour générer des embeddings textuels via Gemini"""

    def __init__(self, api_key: str = None):
        """
        Initialise le service d'embeddings

        Args:
            api_key: Google API Key (GOOGLE_API_KEY env var si None)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required for embeddings")

        # Embeddings pour documents (task_type=RETRIEVAL_DOCUMENT)
        # Utilisé pour stocker un thème dans la base
        # NOTE: On force explicitement 768 dimensions pour compatibilité pgvector ivfflat
        self.doc_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type="RETRIEVAL_DOCUMENT",
            google_api_key=self.api_key,
            model_kwargs={"output_dimensionality": 768}
        )

        # Embeddings pour queries (task_type=RETRIEVAL_QUERY)
        # Utilisé pour rechercher des thèmes similaires
        self.query_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type="RETRIEVAL_QUERY",
            google_api_key=self.api_key,
            model_kwargs={"output_dimensionality": 768}
        )

        logger.info("EmbeddingService initialized (gemini-embedding-001, 768D)")

    def embed_document(self, text: str) -> List[float]:
        """
        Génère un embedding pour un document (768D)

        Utilisé lors de la création/mise à jour d'un thème
        pour le stocker dans pgvector.

        Args:
            text: Texte du document à embedder

        Returns:
            Vecteur de 768 dimensions (liste de floats)
        """
        try:
            vector = self.doc_embeddings.embed_query(text)
            logger.debug(f"Document embedded: {len(text)} chars → {len(vector)}D vector")
            return vector
        except Exception as e:
            logger.error(f"Erreur embed_document: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Génère un embedding pour une requête de recherche (768D)

        Utilisé lors de la recherche RAG pour trouver
        les thèmes les plus similaires.

        Args:
            text: Texte de la requête

        Returns:
            Vecteur de 768 dimensions (liste de floats)
        """
        try:
            vector = self.query_embeddings.embed_query(text)
            logger.debug(f"Query embedded: {len(text)} chars → {len(vector)}D vector")
            return vector
        except Exception as e:
            logger.error(f"Erreur embed_query: {e}")
            raise

    def embed_documents_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Génère des embeddings pour plusieurs documents (batch)

        Utilisé pour le backfill ou le traitement batch.
        Plus efficace que des appels individuels.

        Args:
            texts: Liste de textes à embedder

        Returns:
            Liste de vecteurs 768D
        """
        try:
            vectors = self.doc_embeddings.embed_documents(texts)
            logger.info(f"Batch embedded: {len(texts)} documents → {len(vectors)} vectors")
            return vectors
        except Exception as e:
            logger.error(f"Erreur embed_documents_batch: {e}")
            raise
