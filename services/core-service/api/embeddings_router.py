"""
API Router pour génération d'embeddings
Centralise les embeddings Gemini pour Gateway et App Services
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List
import logging

from core.embeddings.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/embed", tags=["Embeddings"])

# Singleton service (initialisé au démarrage)
_embedding_service: EmbeddingService = None


def get_embedding_service() -> EmbeddingService:
    """Récupère ou crée le service d'embeddings"""
    global _embedding_service
    if _embedding_service is None:
        try:
            _embedding_service = EmbeddingService()
        except ValueError as e:
            logger.error(f"Impossible d'initialiser EmbeddingService: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GOOGLE_API_KEY not configured"
            )
    return _embedding_service


# === Pydantic Models ===

class EmbedRequest(BaseModel):
    """Requête pour générer un embedding"""
    text: str = Field(..., min_length=1, max_length=5000, description="Texte à embedder")
    task_type: str = Field(
        default="document",
        description="Type: 'document' (stockage) ou 'query' (recherche)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Analyse des fournisseurs de batteries lithium-ion",
                "task_type": "query"
            }
        }


class EmbedBatchRequest(BaseModel):
    """Requête pour générer plusieurs embeddings (batch)"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="Liste de textes")

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Premier document",
                    "Deuxième document",
                    "Troisième document"
                ]
            }
        }


class EmbedResponse(BaseModel):
    """Réponse avec un embedding"""
    embedding: List[float] = Field(..., description="Vecteur 768D")
    dimensions: int = Field(..., description="Nombre de dimensions (768)")
    model: str = Field(default="gemini-embedding-001", description="Modèle utilisé")


class EmbedBatchResponse(BaseModel):
    """Réponse avec plusieurs embeddings"""
    embeddings: List[List[float]] = Field(..., description="Liste de vecteurs 768D")
    count: int = Field(..., description="Nombre d'embeddings générés")
    dimensions: int = Field(..., description="Dimensions par vecteur (768)")
    model: str = Field(default="gemini-embedding-001", description="Modèle utilisé")


# === Endpoints ===

@router.post("", response_model=EmbedResponse, summary="Generate Embedding")
async def generate_embedding(request: EmbedRequest):
    """
    Génère un embedding pour un texte via Gemini

    **Task types:**
    - `document`: Pour stocker un document (thème, texte) dans la base
      - Utilise RETRIEVAL_DOCUMENT
      - Optimisé pour le stockage
    - `query`: Pour rechercher des documents similaires
      - Utilise RETRIEVAL_QUERY
      - Optimisé pour la recherche RAG

    **Output:**
    - Vecteur de 768 dimensions (float32)
    - Compatible pgvector: `vector(768)`

    **Coût:** Gratuit (Gemini Embeddings)
    """
    try:
        service = get_embedding_service()

        # Générer l'embedding selon le task_type
        if request.task_type == "query":
            vector = service.embed_query(request.text)
        else:
            vector = service.embed_document(request.text)

        return EmbedResponse(
            embedding=vector,
            dimensions=len(vector)
        )

    except ValueError as e:
        # Configuration manquante
        logger.error(f"Configuration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        # Erreur API Gemini
        logger.error(f"Erreur génération embedding: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur génération embedding: {str(e)}"
        )


@router.post("/batch", response_model=EmbedBatchResponse, summary="Generate Batch Embeddings")
async def generate_batch_embeddings(request: EmbedBatchRequest):
    """
    Génère des embeddings pour plusieurs textes (batch processing)

    **Utilisation:**
    - Backfill de thèmes existants
    - Traitement batch de documents
    - Plus efficace que des appels individuels

    **Limites:**
    - Maximum 100 textes par requête
    - Chaque texte max 5000 caractères

    **Output:**
    - Liste de vecteurs 768D
    - Ordre préservé (vectors[i] correspond à texts[i])

    **Coût:** Gratuit (Gemini Embeddings)
    """
    try:
        service = get_embedding_service()

        # Générer tous les embeddings en batch
        vectors = service.embed_documents_batch(request.texts)

        return EmbedBatchResponse(
            embeddings=vectors,
            count=len(vectors),
            dimensions=len(vectors[0]) if vectors else 0
        )

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Erreur génération batch embeddings: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur génération batch embeddings: {str(e)}"
        )


@router.get("/health", summary="Health Check")
async def health_check():
    """
    Vérifie que le service d'embeddings est opérationnel

    Retourne:
    - status: "ok" si le service est disponible
    - model: Nom du modèle utilisé
    """
    try:
        service = get_embedding_service()
        return {
            "status": "ok",
            "model": "gemini-embedding-001",
            "dimensions": 768,
            "task_types": ["document", "query"]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
