"""
Standard event names used across services.
"""


class EventNames:
    """
    Standard event names used across services.

    Using constants helps avoid typos and makes it easier to find
    all event publishers and subscribers.
    """

    # Captation events
    CAPTATION_STARTED = "captation.started"
    CAPTATION_PROGRESS = "captation.progress"
    CAPTATION_COMPLETED = "captation.completed"
    CAPTATION_FAILED = "captation.failed"

    # Analysis events
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_PROGRESS = "analysis.progress"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"

    # Batch report events
    BATCH_STARTED = "batch.started"
    BATCH_PROGRESS = "batch.progress"
    BATCH_STORE_COMPLETED = "batch.store_completed"
    BATCH_STORE_FAILED = "batch.store_failed"
    BATCH_COMPLETED = "batch.completed"
    BATCH_FAILED = "batch.failed"

    # LLM events
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_ERROR = "llm.error"

    # System events
    SERVICE_STARTED = "service.started"
    SERVICE_STOPPED = "service.stopped"
    SERVICE_HEALTH_DEGRADED = "service.health_degraded"

