"""
Queued request model for LLM queue service
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

from .llm_request import LLMRequest
from .request_priority import RequestPriority


@dataclass
class QueuedRequest:
    """Requête LLM en queue avec métadonnées"""
    request: LLMRequest
    priority: RequestPriority = RequestPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    attempt: int = 0
    max_retries: int = 3
    future: asyncio.Future = field(default_factory=asyncio.Future)

    def __lt__(self, other: "QueuedRequest") -> bool:
        """Tri par priorité puis par timestamp pour la queue de priorité"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at

