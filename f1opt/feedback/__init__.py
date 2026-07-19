"""Driver feedback engine.

Re-exports the public entry point :func:`generate_feedback`, the
:class:`FeedbackEngine` wrapper, and the :data:`FEEDBACK_DIMENSIONS` constant
(all 10 spec dimensions). The engine produces evidence-grounded feedback
covering every dimension (balance / grip / tyres / braking / ers_drs /
throttle_brake_smoothness / confidence / lap_time_potential / sector_compare /
setup_advice) and supports a pluggable chat backend.

Re-exports :class:`ConversationSession` and the session registry helpers
(:func:`get_session`, :func:`reset_sessions`) for multi-turn driver dialogue.
"""

from __future__ import annotations

from .conversation import (
    ConversationMemory,
    ConversationSession,
    conversation_token_usage,
    estimate_tokens,
    get_session,
    reset_sessions,
    trim_conversation_history,
)
from .engine import (
    FEEDBACK_DIMENSIONS,
    FeedbackEngine,
    generate_feedback,
    generate_feedback_stream,
    generate_feedback_stream_async,
)
from .intent import IntentResult, classify_intent
from .language import (
    LanguageAdapter,
    adapt_response_language,
    detect_language,
    get_system_prompt_for_language,
)
from .quality import ResponseQualityReport, assess_response_quality

__all__ = [
    "ConversationMemory",
    "ConversationSession",
    "conversation_token_usage",
    "estimate_tokens",
    "FEEDBACK_DIMENSIONS",
    "FeedbackEngine",
    "IntentResult",
    "LanguageAdapter",
    "ResponseQualityReport",
    "adapt_response_language",
    "assess_response_quality",
    "classify_intent",
    "detect_language",
    "generate_feedback",
    "generate_feedback_stream",
    "generate_feedback_stream_async",
    "get_session",
    "get_system_prompt_for_language",
    "reset_sessions",
    "trim_conversation_history",
]
