"""
LLM Queue Service - Intelligent queuing system for LLM requests with concurrency control and retry logic
"""

import asyncio
import logging
import time
from typing import Any

from .llm_request import LLMRequest
from .llm_response import LLMResponse
from .llm_service import LLMService
from .providers.llm_provider_error import LLMProviderError
from .providers.llm_provider_timeout_error import LLMProviderTimeoutError
from .queued_request import QueuedRequest
from .request_priority import RequestPriority

logger = logging.getLogger(__name__)


class LLMQueueService:
    """Service de queue intelligent pour les requêtes LLM"""

    def __init__(self, max_concurrent: int = 2, max_retries: int = 3):
        """
        Initialise le service de queue LLM

        Args:
            max_concurrent: Nombre maximum de requêtes LLM simultanées
            max_retries: Nombre maximum de tentatives par requête
        """
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries

        # Queue de priorité pour les requêtes
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

        # Semaphore pour limiter la concurrence
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Service LLM sous-jacent
        self._llm_service = LLMService()

        # Workers pour traiter les requêtes
        self._workers = []
        self._running = False

        # Métriques
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retried_requests": 0,
            "queue_size": 0,
            "active_workers": 0,
            "avg_processing_time": 0.0
        }

        logger.info(f"LLMQueueService initialized with {max_concurrent} max concurrent workers, {max_retries} max retries")

    async def start(self):
        """Démarre les workers de traitement"""
        if self._running:
            return

        self._running = True

        # Démarrer les workers
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)

        logger.info(f"Started {len(self._workers)} LLM queue workers")

    async def stop(self):
        """Arrête les workers de traitement"""
        if not self._running:
            return

        self._running = False

        # Arrêter tous les workers
        for worker in self._workers:
            worker.cancel()

        # Attendre que tous se terminent
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        logger.info("Stopped all LLM queue workers")

    async def enqueue_request(
        self,
        request: LLMRequest,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 300.0
    ) -> LLMResponse:
        """
        Ajoute une requête à la queue et attend la réponse

        Args:
            request: Requête LLM à traiter
            priority: Priorité de la requête
            timeout: Timeout en secondes pour la requête complète

        Returns:
            Réponse LLM

        Raises:
            asyncio.TimeoutError: Si la requête timeout
            LLMProviderError: Si toutes les tentatives échouent
        """
        # Démarrer le service si ce n'est pas déjà fait (fallback si lifespan n'a pas démarré)
        if not self._running:
            logger.warning("LLM Queue Service not started, starting now...")
            await self.start()

        # Créer la requête queueable
        queued_request = QueuedRequest(
            request=request,
            priority=priority,
            max_retries=self.max_retries
        )

        # Ajouter à la queue
        await self._queue.put(queued_request)
        self._metrics["total_requests"] += 1
        self._metrics["queue_size"] = self._queue.qsize()

        logger.info(f"Enqueued LLM request (priority: {priority.name}, queue size: {self._queue.qsize()})")

        # Attendre la réponse avec timeout
        try:
            return await asyncio.wait_for(queued_request.future, timeout=timeout)
        except TimeoutError:
            logger.error(f"LLM request timed out after {timeout}s")
            raise

    async def _worker(self, worker_id: str):
        """
        Worker qui traite les requêtes de la queue

        Args:
            worker_id: Identifiant du worker
        """
        logger.info(f"LLM worker {worker_id} started")

        while self._running:
            try:
                # Récupérer une requête de la queue
                queued_request = await self._queue.get()
                self._metrics["queue_size"] = self._queue.qsize()

                logger.info(f"Worker {worker_id} processing request (attempt {queued_request.attempt + 1}/{queued_request.max_retries + 1})")

                # Traiter la requête avec retry
                await self._process_request_with_retry(queued_request, worker_id)

                # Marquer comme terminé
                self._queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                # Continuer le traitement même en cas d'erreur

        logger.info(f"LLM worker {worker_id} stopped")

    async def _process_request_with_retry(self, queued_request: QueuedRequest, worker_id: str):
        """
        Traite une requête avec retry automatique

        Args:
            queued_request: Requête à traiter
            worker_id: ID du worker qui traite
        """
        start_time = time.time()

        # Acquérir le semaphore pour limiter la concurrence
        async with self._semaphore:
            self._metrics["active_workers"] += 1

            try:
                # Boucle de retry: jusqu'à max_retries tentatives (1 initiale + max_retries retries)
                # attempt commence à 0, donc on fait attempt + 1 tentatives au total
                while queued_request.attempt < queued_request.max_retries:
                    try:
                        current_attempt = queued_request.attempt + 1
                        logger.info(f"Worker {worker_id} attempting LLM request (attempt {current_attempt}/{queued_request.max_retries})")

                        # Traiter avec le service LLM
                        response = await self._llm_service.generate(queued_request.request)

                        # Succès !
                        processing_time = time.time() - start_time
                        self._update_success_metrics(processing_time)

                        logger.info(f"✅ Worker {worker_id} completed request successfully in {processing_time:.2f}s (attempt {current_attempt})")

                        # Retourner le résultat via le Future
                        if not queued_request.future.done():
                            queued_request.future.set_result(response)
                        return

                    except (LLMProviderTimeoutError, LLMProviderError) as e:
                        queued_request.attempt += 1
                        current_attempt = queued_request.attempt

                        if current_attempt <= queued_request.max_retries:
                            # Backoff exponentiel: 2^(attempt-1) secondes (1s, 2s, 4s, etc.)
                            backoff_time = min(2 ** (current_attempt - 1), 30)  # Max 30s

                            logger.warning(f"⚠️ Worker {worker_id} attempt {current_attempt} failed: {str(e)}")
                            logger.info(f"🔄 Worker {worker_id} retrying in {backoff_time}s... (remaining attempts: {queued_request.max_retries - current_attempt})")

                            self._metrics["retried_requests"] += 1
                            await asyncio.sleep(backoff_time)
                        else:
                            # Toutes les tentatives échouées
                            logger.error(f"❌ Worker {worker_id} all {queued_request.max_retries} attempts failed: {str(e)}")
                            self._metrics["failed_requests"] += 1

                            if not queued_request.future.done():
                                queued_request.future.set_exception(e)
                            return

                    except Exception as e:
                        # Erreur non-retriable (pas un LLMProviderError)
                        logger.error(f"❌ Worker {worker_id} non-retriable error (attempt {queued_request.attempt + 1}): {str(e)}")
                        logger.exception(e)  # Stack trace complète
                        self._metrics["failed_requests"] += 1

                        if not queued_request.future.done():
                            queued_request.future.set_exception(e)
                        return

                # Si on sort de la boucle sans succès, toutes les tentatives sont épuisées
                if not queued_request.future.done():
                    error_msg = f"All {queued_request.max_retries} attempts exhausted without success"
                    logger.error(f"❌ Worker {worker_id}: {error_msg}")
                    queued_request.future.set_exception(LLMProviderError(message=error_msg))

            finally:
                self._metrics["active_workers"] -= 1

    def _update_success_metrics(self, processing_time: float):
        """Met à jour les métriques de succès"""
        self._metrics["successful_requests"] += 1

        # Calcul de la moyenne mobile simple
        total_requests = self._metrics["successful_requests"]
        current_avg = self._metrics["avg_processing_time"]

        # Moyenne mobile: new_avg = (old_avg * (n-1) + new_value) / n
        self._metrics["avg_processing_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )

    def get_metrics(self) -> dict[str, Any]:
        """Retourne les métriques du service"""
        return {
            **self._metrics,
            "queue_size": self._queue.qsize(),
            "is_running": self._running,
            "max_concurrent": self.max_concurrent,
            "max_retries": self.max_retries
        }

    def get_health(self) -> dict[str, Any]:
        """Retourne l'état de santé du service"""
        metrics = self.get_metrics()

        # Calcul du taux de succès
        total = metrics["successful_requests"] + metrics["failed_requests"]
        success_rate = (metrics["successful_requests"] / total * 100) if total > 0 else 100

        # Détermination du statut
        if not self._running:
            status = "stopped"
        elif metrics["queue_size"] > 50:  # Seuil arbitraire
            status = "overloaded"
        elif success_rate < 80:  # Seuil arbitraire
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "queue_size": metrics["queue_size"],
            "active_workers": metrics["active_workers"],
            "success_rate": round(success_rate, 2),
            "avg_processing_time": round(metrics["avg_processing_time"], 2),
            "total_requests": total
        }
