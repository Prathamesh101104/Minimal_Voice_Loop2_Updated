import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipecat.frames.frames import (
    LLMContextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from rag import get_rag
from semantic_cache import get_semantic_cache
from bot import (
    SemanticRAGProcessor,
    InterruptionAwareHangupProcessor,
    CacheHitFrame,
    is_closing_intent,
    _THANK_YOU_RAW_AUDIO,
    _THANK_YOU_TEXT,
)

async def test_closing_flow():
    print("Testing pre-recorded thank you & closing flow...")
    rag = get_rag()
    cache = get_semantic_cache()
    hangup_processor = InterruptionAwareHangupProcessor(hangup_delay=5.0)
    processor = SemanticRAGProcessor(rag, cache, hangup_processor=hangup_processor)

    # Mock downstream capture
    captured_frames = []

    async def mock_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
        captured_frames.append(frame)

    processor.push_frame = mock_push_frame

    # Test closing query
    closing_query = "Okay. That's it. Thank you."
    assert is_closing_intent(closing_query), f"Failed to detect closing intent for: {closing_query}"

    context = LLMContext(
        messages=[
            {"role": "user", "content": closing_query}
        ]
    )
    frame = LLMContextFrame(context)

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    # Verify assistant message was added
    assert len(context.messages) == 2, f"Expected 2 messages, got {len(context.messages)}"
    assert context.messages[-1]["content"] == _THANK_YOU_TEXT

    # Verify hangup task was armed
    assert hangup_processor._pending_hangup_task is not None, "Hangup task was not armed!"

    # Verify captured frames
    frame_types = [type(f).__name__ for f in captured_frames]
    print(f"Captured frame sequence: {frame_types}")

    assert "CacheHitFrame" in frame_types, "CacheHitFrame missing"
    assert "TTSAudioRawFrame" in frame_types, "TTSAudioRawFrame missing"
    assert "TTSStartedFrame" in frame_types, "TTSStartedFrame missing"
    assert "TTSStoppedFrame" in frame_types, "TTSStoppedFrame missing"

    raw_frame = next(f for f in captured_frames if isinstance(f, TTSAudioRawFrame))
    assert raw_frame.sample_rate == 8000, f"Expected sample rate 8000, got {raw_frame.sample_rate}"
    assert len(raw_frame.audio) == len(_THANK_YOU_RAW_AUDIO)

    # Cancel hangup task to clean up
    hangup_processor._pending_hangup_task.cancel()
    print("✅ Pre-recorded thank-you playback & hangup test: [PASS]")

if __name__ == "__main__":
    asyncio.run(test_closing_flow())
