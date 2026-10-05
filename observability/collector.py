"""observability/collector.py

Pipeline FrameProcessor that tracks turn-by-turn latency telemetry, transcripts, and usage metrics.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFAMetricsData,
    TTFBMetricsData,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from evals.store import save_call_metrics


class CallTelemetryCollector(FrameProcessor):
    """Intercepts and records granular turn-by-turn metrics and transcripts during a call."""

    def __init__(self, call_id: str, context: Any = None):
        super().__init__()
        self.call_id = call_id
        self.context = context
        self.start_time = time.time()
        self.turns: List[Dict[str, Any]] = []
        self._current_turn: Optional[Dict[str, Any]] = None
        self._turn_index = 0
        self._tokens = {"prompt": 0, "completion": 0}
        self._is_saved = False

    def _get_or_create_turn(self) -> Dict[str, Any]:
        if self._current_turn is None:
            self._turn_index += 1
            self._current_turn = {
                "turn_id": self._turn_index,
                "user_query": "",
                "agent_text": "",
                "ttfa_ms": 0.0,
                "stt_latency_ms": 0.0,
                "llm_ttft_ms": 0.0,
                "tts_latency_ms": 0.0,
                "cache_hit": False,
                "cache_similarity": 0.0,
            }
        return self._current_turn

    def _finalize_current_turn(self):
        if self._current_turn is not None:
            # Fallback: if user_query is empty, extract latest user query from context
            if not self._current_turn.get("user_query") and self.context:
                try:
                    msgs = getattr(self.context, "messages", []) or []
                    user_msgs = [m.get("content", "") for m in msgs if m.get("role") == "user" and m.get("content")]
                    if user_msgs:
                        self._current_turn["user_query"] = user_msgs[-1]
                except Exception:
                    pass

            # Fallback: if ttfa_ms is 0, calculate estimated TTFA from pipeline latencies
            if self._current_turn.get("ttfa_ms", 0.0) == 0.0 and self._current_turn.get("llm_ttft_ms", 0.0) > 0.0:
                self._current_turn["ttfa_ms"] = round(
                    self._current_turn.get("stt_latency_ms", 0.0)
                    + self._current_turn.get("llm_ttft_ms", 0.0)
                    + self._current_turn.get("tts_latency_ms", 0.0),
                    1,
                )

            # Only save if there was actual interaction (query or agent text or ttfa)
            if (
                self._current_turn.get("user_query")
                or self._current_turn.get("agent_text")
                or self._current_turn.get("ttfa_ms", 0) > 0
            ):
                self.turns.append(self._current_turn)
            self._current_turn = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # 1. User started speaking -> transition to a new turn
        if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            if self._current_turn and (self._current_turn.get("agent_text") or self._current_turn.get("ttfa_ms", 0) > 0):
                self._finalize_current_turn()

        # 2. Transcription Frame -> captures user query
        elif isinstance(frame, TranscriptionFrame):
            if frame.text and frame.text.strip():
                turn = self._get_or_create_turn()
                if turn["user_query"]:
                    turn["user_query"] += " " + frame.text.strip()
                else:
                    turn["user_query"] = frame.text.strip()

        # 2b. Cache Hit Frame -> marks turn as resolved instantly from semantic cache (0ms LLM)
        elif hasattr(frame, "similarity") and hasattr(frame, "canonical_query"):
            turn = self._get_or_create_turn()
            turn["cache_hit"] = True
            turn["cache_similarity"] = getattr(frame, "similarity", 1.0)
            turn["llm_ttft_ms"] = 0.0
            if getattr(frame, "query", None) and not turn["user_query"]:
                turn["user_query"] = frame.query
            if getattr(frame, "response", None) and not turn["agent_text"]:
                turn["agent_text"] = frame.response

        # 3. Agent speech text frames -> captures agent response
        elif isinstance(frame, (TTSTextFrame, LLMTextFrame, TTSSpeakFrame, TextFrame)):
            text = getattr(frame, "text", "")
            if text and text.strip():
                turn = self._get_or_create_turn()
                if not turn["cache_hit"]:
                    turn["agent_text"] += text

        # 4. Metrics Frames -> captures granular latencies & tokens
        elif isinstance(frame, MetricsFrame):
            turn = self._get_or_create_turn()
            for metric in frame.data:
                # Time-to-First-Audio (TTFA)
                if isinstance(metric, TTFAMetricsData):
                    if hasattr(metric, "ttfa") and metric.ttfa:
                        turn["ttfa_ms"] = round(metric.ttfa * 1000.0, 1)

                # Time-to-First-Byte (LLM TTFT or TTS Latency)
                elif isinstance(metric, TTFBMetricsData):
                    processor_name = (metric.processor or "").lower()
                    val_ms = round(metric.value * 1000.0, 1)
                    if "llm" in processor_name or "groq" in processor_name or "openai" in processor_name:
                        turn["llm_ttft_ms"] = val_ms
                    elif "tts" in processor_name or "sarvam" in processor_name:
                        turn["tts_latency_ms"] = val_ms

                # Processing time metrics (e.g. STT)
                elif isinstance(metric, ProcessingMetricsData):
                    processor_name = (metric.processor or "").lower()
                    val_ms = round(metric.value * 1000.0, 1)
                    if "stt" in processor_name or "deepgram" in processor_name:
                        turn["stt_latency_ms"] = val_ms

                # Token usage
                elif isinstance(metric, LLMUsageMetricsData):
                    if hasattr(metric, "value") and metric.value:
                        self._tokens["prompt"] += getattr(metric.value, "prompt_tokens", 0) or 0
                        self._tokens["completion"] += getattr(metric.value, "completion_tokens", 0) or 0

        # Pass frame along pipeline
        await self.push_frame(frame, direction)

    def finalize(self, status: str = "completed"):
        """Save call scorecard and telemetry."""
        if self._is_saved:
            return
        self._finalize_current_turn()
        duration = time.time() - self.start_time
        try:
            record = save_call_metrics(
                call_id=self.call_id,
                turns=self.turns,
                duration_seconds=duration,
                status=status,
                tokens=self._tokens,
            )
            self._is_saved = True
            logger.info(
                f"📊 [Telemetry] Saved metrics for call {self.call_id}: "
                f"{len(self.turns)} turns, avg TTFA: {record['summary']['avg_ttfa_ms']}ms"
            )
        except Exception as e:
            logger.error(f"Failed to save metrics for call {self.call_id}: {e}")
