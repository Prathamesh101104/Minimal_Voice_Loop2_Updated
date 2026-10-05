#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import os
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import aiohttp
from typing import Dict, List, Optional, Set, Tuple
from dotenv import load_dotenv
from fastapi import WebSocketDisconnect
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.serializers.vobiz import (
    HANGUP_BOTH,
    VobizFrameSerializer,
    parse_vobiz_start,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.sarvam.tts import SarvamHttpTTSService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

load_dotenv(override=True)


# ── Nimbus Knowledge Base (SimpleRAG Engine & Semantic Cache) ──
from rag import SimpleRAG, get_rag
from semantic_cache import CacheHitFrame, SemanticResponseCache, get_semantic_cache

rag = get_rag()
semantic_cache = get_semantic_cache()
_NIMBUS_DATA_DIR = Path(__file__).resolve().parent / "nimbus-voice-agent-starter" / "data"
_CONSOLIDATED_PATH = _NIMBUS_DATA_DIR / "nimbus_consolidated_knowledge.json"
if _CONSOLIDATED_PATH.exists():
    _ck = json.loads(_CONSOLIDATED_PATH.read_text(encoding="utf-8"))
    _co = _ck.get("company", {})
    _pol = _ck.get("policies", {})
    NIMBUS_COMPANY_INFO = (
        f"Nimbus Company Profile & Policies:\n"
        f"- Headquarters: {_co.get('hq', 'Austin, TX')}, Founded: {_co.get('founded', '2014')}\n"
        f"- Overview: {_co.get('about', 'All-in-one cloud business software suite.')}\n"
        f"- Refund Policy: {_pol.get('refund', {}).get('summary', '30-day money-back guarantee on all plans.')}\n"
        f"- Free Trial: {_pol.get('free_trial', {}).get('summary', '14-day free trial on paid plans with no credit card required.')}\n"
        f"- Billing: {_pol.get('billing', {}).get('summary', 'Monthly or annual billing with 20% annual discount.')}\n"
        f"- Cancellation: {_pol.get('cancellation', {}).get('summary', 'Cancel anytime in admin dashboard without penalty.')}\n"
        f"- Support: {_pol.get('support', {}).get('summary', '24/7 email and chat support.')}\n"
        f"- Security & Compliance: SOC 2 Type II certified, ISO 27001 compliant, GDPR and CCPA compliant. TLS 1.3 in transit, AES-256 at rest.\n"
        f"- Data Residency: Regional data centers available in the United States, European Union (Frankfurt), India (IN), and Australia (AU).\n"
        f"- SLA: 99.9% uptime for standard plans, 99.99% for enterprise."
    )
else:
    NIMBUS_CONTEXT = (_NIMBUS_DATA_DIR / "context.md").read_text(encoding="utf-8")
    _CONTEXT_SECTIONS = NIMBUS_CONTEXT.split("# Product Catalog")
    NIMBUS_COMPANY_INFO = _CONTEXT_SECTIONS[0].strip()

# ── Pre-recorded Greeting Audio (Instant Playback on Connect) ──
_GREETING_WAV_PATH = Path(__file__).resolve().parent / "assets" / "greeting_8k.wav"
_GREETING_RAW_AUDIO: bytes = b""

if _GREETING_WAV_PATH.exists():
    try:
        import wave
        with wave.open(str(_GREETING_WAV_PATH), "rb") as _w:
            _GREETING_RAW_AUDIO = _w.readframes(_w.getnframes())
            logger.info(f"Loaded pre-recorded greeting audio: {len(_GREETING_RAW_AUDIO)} bytes (8000 Hz)")
    except Exception as _e:
        logger.warning(f"Could not load pre-recorded greeting audio: {_e}")

# ── Pre-recorded Thank You Audio (Instant Playback on Closing / Thank You) ──
_THANK_YOU_WAV_PATH = Path(__file__).resolve().parent / "assets" / "thank_you_8k.wav"
_THANK_YOU_RAW_AUDIO: bytes = b""
_THANK_YOU_TEXT = "Thank you for calling Nimbus. Have a wonderful day! Goodbye."

if _THANK_YOU_WAV_PATH.exists():
    try:
        import wave
        with wave.open(str(_THANK_YOU_WAV_PATH), "rb") as _w:
            _THANK_YOU_RAW_AUDIO = _w.readframes(_w.getnframes())
            logger.info(f"Loaded pre-recorded thank you audio: {len(_THANK_YOU_RAW_AUDIO)} bytes (8000 Hz)")
    except Exception as _e:
        logger.warning(f"Could not load pre-recorded thank you audio: {_e}")


# ── Closing Intent Classifier (Detects 'thank you', 'that's it', 'all good', etc.) ──
_CLOSING_PRODUCT_AND_POLICY_KEYWORDS: Set[str] = {
    "crm", "sso", "vault", "leads", "recruit", "payroll", "books", "invoice",
    "expense", "people", "desk", "chat", "projects", "docs", "analytics",
    "datapipe", "endpoint", "dashboards", "boards", "knowledge", "platform",
    "sites", "campaigns", "quote", "refund", "trial", "pricing", "price",
    "cost", "sla", "security", "support", "billing", "payment", "residency"
}

_CLOSING_QUESTION_PHRASES: List[str] = [
    r"\b(what|how|why|when|where|who|which)\b",
    r"\b(can you|could you|tell me|explain|help me with)\b",
    r"\b(how much|how many|is there|are there|do you have)\b",
]

_CLOSING_PATTERNS: List[str] = [
    r"\b(that'?s\s+(?:it|all|everything))\b",
    r"\b(that\s+is\s+(?:it|all|everything))\b",
    r"\b(that\'?ll\s+be\s+all)\b",
    r"\b(nothing\s+else)\b",
    r"\b(no\s+more\s+questions)\b",
    r"\b(i\'?m\s+(?:all\s+set|good|done))\b",
    r"\b(all\s+good|all\s+set)\b",
    r"\b(thank\s+you|thanks|thx)\b",
    r"\b(bye|goodbye|have\s+a\s+good\s+(?:day|one))\b",
]

def is_closing_intent(query: str) -> bool:
    """Identify if user query is a signoff/closing statement like 'that's it', 'thank you', etc."""
    clean = query.strip().lower()
    if not clean:
        return False

    # 1. Any product/policy keywords -> definitely NOT closing
    words = set(re.findall(r"\b[a-z0-9\']+\b", clean))
    if words & _CLOSING_PRODUCT_AND_POLICY_KEYWORDS:
        return False

    # 2. Any explicit question phrasing -> definitely NOT closing
    for q_pat in _CLOSING_QUESTION_PHRASES:
        if re.search(q_pat, clean):
            return False

    # 3. Check for closing intent pattern
    for pat in _CLOSING_PATTERNS:
        if re.search(pat, clean):
            return True

    return False


class SemanticRAGProcessor(FrameProcessor):
    """Semantic Response Cache & RAG processor.

    1. Checks incoming user queries against SemanticResponseCache using vector embeddings & cosine similarity.
    2. On Cache HIT:
       - Skips RAG database retrieval & LLM generation completely!
       - Appends assistant message to context.messages for multi-turn conversation memory.
       - Pushes CacheHitFrame, LLMFullResponseStartFrame(), TextFrame(cached_text), and LLMFullResponseEndFrame()
         downstream to TTS & CallTelemetryCollector.
       - Drops the LLMContextFrame so the LLM is never invoked (sub-5ms response).
    3. On Cache MISS:
       - Fallback to SimpleRAG retrieval to enrich user prompt with retrieved website knowledge.
       - Forwards LLMContextFrame to LLM (Google Gemini).
       - Buffers subsequent LLM TextFrames to dynamically store and learn new responses into the cache!
    """

    def __init__(
        self,
        rag_engine: SimpleRAG,
        cache: Optional[SemanticResponseCache] = None,
        hangup_processor: Optional[InterruptionAwareHangupProcessor] = None,
    ):
        super().__init__()
        self._rag = rag_engine
        self._cache = cache or semantic_cache
        self._hangup_processor = hangup_processor
        self._pending_user_query: Optional[str] = None
        self._llm_response_buffer: List[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            msgs = frame.context.messages
            user_text = ""
            user_msg = None
            for i in range(len(msgs) - 1, -1, -1):
                m = msgs[i]
                if m.get("role") == "user" and m.get("content"):
                    user_text = m["content"]
                    user_msg = m
                    break

            if user_text:
                # Extract clean user query
                clean_query = user_text.split("\n\n[Retrieved Website Knowledge]:")[0].strip()

                # ── 0. Check Pre-recorded Closing / Thank You Intent ──
                if is_closing_intent(clean_query):
                    logger.info(
                        f"⚡ [ClosingIntent] Detected closing query: '{clean_query}'. "
                        f"Playing pre-recorded thank you audio & arming hangup!"
                    )
                    # Update context messages for conversation history continuity
                    msgs.append({"role": "assistant", "content": _THANK_YOU_TEXT})

                    # Trigger interruption-aware hangup timer (5.0s for audio playback + buffer)
                    if self._hangup_processor:
                        self._hangup_processor.trigger_hangup()

                    # Push telemetry frame
                    await self.push_frame(
                        CacheHitFrame(
                            query=clean_query,
                            response=_THANK_YOU_TEXT,
                            similarity=1.0,
                            canonical_query="Thank you for calling Nimbus. Have a wonderful day! Goodbye.",
                            category="closing",
                        ),
                        direction,
                    )

                    if _THANK_YOU_RAW_AUDIO:
                        # Direct raw audio playback: 0ms LLM, 0ms TTS latency!
                        await self.push_frame(TTSStartedFrame(), direction)
                        await self.push_frame(
                            TTSAudioRawFrame(
                                audio=_THANK_YOU_RAW_AUDIO,
                                sample_rate=8000,
                                num_channels=1,
                            ),
                            direction,
                        )
                        await self.push_frame(TTSStoppedFrame(), direction)
                    else:
                        # Fallback if audio file missing
                        await self.push_frame(LLMFullResponseStartFrame(), direction)
                        await self.push_frame(TextFrame(text=_THANK_YOU_TEXT), direction)
                        await self.push_frame(LLMFullResponseEndFrame(), direction)

                    # Completely drop LLMContextFrame so the LLM is skipped!
                    return

                # ── 1. Check Semantic Cache ──
                cached = self._cache.get(clean_query)
                if cached:
                    cached_text, sim, canonical_query = cached
                    logger.info(
                        f"⚡ [SemanticCache] CACHE HIT (sim={sim:.2f}): '{clean_query}' -> '{canonical_query}' (Skipping RAG & LLM)"
                    )

                    # Update context messages for conversation history continuity
                    msgs.append({"role": "assistant", "content": cached_text})

                    # Emit CacheHitFrame for telemetry and LLM frames to trigger TTS directly
                    await self.push_frame(
                        CacheHitFrame(
                            query=clean_query,
                            response=cached_text,
                            similarity=sim,
                            canonical_query=canonical_query,
                            category=getattr(cached, "category", "policy"),
                        ),
                        direction,
                    )
                    await self.push_frame(LLMFullResponseStartFrame(), direction)
                    await self.push_frame(TextFrame(text=cached_text), direction)
                    await self.push_frame(LLMFullResponseEndFrame(), direction)

                    # Completely drop LLMContextFrame so the LLM is skipped!
                    return

                # ── 2. Cache MISS -> Fallback to SimpleRAG ──
                self._pending_user_query = clean_query
                self._llm_response_buffer = []

                if user_msg and "[Retrieved Website Knowledge]:" not in user_text:
                    context_block = self._rag.format_context_for_prompt(clean_query, top_k=2)
                    if context_block:
                        logger.info(f"📚 [SimpleRAG] Injected knowledge for query: '{clean_query}'")
                        user_msg["content"] = (
                            f"{clean_query}\n\n"
                            f"{context_block}\n\n"
                            "Instructions: Answer the caller's question concisely using the retrieved knowledge above."
                        )

            await self.push_frame(frame, direction)
            return

        # ── 3. Dynamic Learning: Buffer LLM response frames on Cache Miss ──
        if isinstance(frame, (TextFrame, LLMTextFrame)) and not isinstance(
            frame, (TranscriptionFrame, InterimTranscriptionFrame)
        ):
            if self._pending_user_query:
                self._llm_response_buffer.append(getattr(frame, "text", ""))
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            if self._pending_user_query and self._llm_response_buffer:
                full_resp = "".join(self._llm_response_buffer).strip()
                # Dynamically cache conversational responses (ignore error or goodbye messages)
                if (
                    len(full_resp) >= 15
                    and not full_resp.startswith("{")
                    and "error" not in full_resp.lower()
                    and "goodbye" not in full_resp.lower()
                ):
                    self._cache.set(self._pending_user_query, full_resp, category="dynamic")
                self._pending_user_query = None
                self._llm_response_buffer = []

            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)


# Backward-compatible alias
SimpleRAGProcessor = SemanticRAGProcessor


from pipecat.audio.utils import create_file_resampler


class FixedVobizFrameSerializer(VobizFrameSerializer):
    """Ensures sample_rate fields are always valid (> 0), ignores CancelFrame,
    and uses a batch/file resampler to prevent chunk tail buffering glitches.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Fix 1: Ensure valid sample rates immediately
        self._vobiz_sample_rate = self._params.vobiz_sample_rate or 8000
        self._sample_rate = self._params.sample_rate or self._vobiz_sample_rate

        # Fix 2: Stream resamplers buffer ~100ms group-delay tail across turns,
        # which causes audio stutter/glitch at the start of subsequent turns.
        # create_file_resampler() resamples whole frames cleanly with 0 leftover buffer.
        self._output_resampler = create_file_resampler()

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, CancelFrame):
            return None
        return await super().serialize(frame)


class InterruptionAwareHangupProcessor(FrameProcessor):
    """Monitors speech interruptions during the final goodbye signoff.

    When the agent bids farewell, this processor arms a delayed hangup timer.
    If the caller interrupts (says 'wait', 'one more question', etc.), the timer
    is cancelled immediately and the assistant continues answering queries.
    """

    def __init__(self, hangup_delay: float = 4.0):
        super().__init__()
        self._hangup_delay = hangup_delay
        self._pending_hangup_task: asyncio.Task | None = None
        self._task: PipelineTask | None = None

    def set_task(self, task: PipelineTask):
        self._task = task

    def trigger_hangup(self):
        """Schedule call termination after signoff speech finishes."""
        if self._pending_hangup_task and not self._pending_hangup_task.done():
            self._pending_hangup_task.cancel()

        async def _delayed_hangup():
            try:
                logger.info(
                    f"📞 [Hangup] Signoff started. Call will end in {self._hangup_delay}s unless interrupted."
                )
                await asyncio.sleep(self._hangup_delay)
                logger.info("📞 [Hangup] Signoff finished without interruption. Ending call now.")
                if self._task:
                    await self._task.queue_frame(EndFrame())
            except asyncio.CancelledError:
                logger.info("🎙️ [Hangup] CANCELLED: Caller interrupted during goodbye! Resuming conversation.")

        self._pending_hangup_task = asyncio.create_task(_delayed_hangup())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Detect interruption during signoff window
        if isinstance(
            frame,
            (
                InterruptionFrame,
                UserStartedSpeakingFrame,
                VADUserStartedSpeakingFrame,
                TranscriptionFrame,
            ),
        ):
            if self._pending_hangup_task and not self._pending_hangup_task.done():
                logger.info(
                    f"🎙️ [Hangup] Detected {type(frame).__name__} during signoff — cancelling hangup!"
                )
                self._pending_hangup_task.cancel()
                self._pending_hangup_task = None

        await self.push_frame(frame, direction)


class EarlyWordAggregator(SimpleTextAggregator):
    """Custom aggregator that emits the first N words of an LLM turn immediately
    to achieve ultra-low Time-To-First-Audio (TTFA) latency, and then aggregates
    the remaining text by full sentences for natural voice cadence.
    """

    def __init__(self, first_words_count: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.first_words_count = first_words_count
        self._first_words_emitted = False
        self._word_buffer = ""

    async def aggregate(self, text: str):
        if not self._first_words_emitted and self.first_words_count > 0:
            for char in text:
                if not self._first_words_emitted:
                    self._word_buffer += char
                    stripped = self._word_buffer.strip()
                    if stripped and char in (" ", "\n", "\t"):
                        words = stripped.split()
                        if len(words) >= self.first_words_count:
                            self._first_words_emitted = True
                            first_chunk = " ".join(words[:self.first_words_count])
                            remainder = " ".join(words[self.first_words_count:])
                            self._text = (remainder + " ") if remainder else ""
                            logger.info(f"⚡ [EarlyWordAggregator] Emitting early {self.first_words_count}-word phrase for fast audio: '{first_chunk}'")
                            yield Aggregation(text=first_chunk, type=AggregationType.SENTENCE)
                            continue
                else:
                    self._text += char
                    result = await self._check_sentence_with_lookahead(char)
                    if result:
                        yield result
        else:
            async for item in super().aggregate(text):
                yield item

    async def flush(self):
        if not self._first_words_emitted and self._word_buffer.strip():
            self._first_words_emitted = True
            res = self._word_buffer.strip()
            self._word_buffer = ""
            self._text = ""
            return Aggregation(text=res, type=AggregationType.SENTENCE)
        self._word_buffer = ""
        return await super().flush()

    async def handle_interruption(self):
        self._first_words_emitted = False
        self._word_buffer = ""
        await super().handle_interruption()

    async def reset(self):
        self._first_words_emitted = False
        self._word_buffer = ""
        await super().reset()


class FixedSarvamHttpTTSService(SarvamHttpTTSService):
    """Sarvam AI HTTP TTS with proper native sample rate labelling, text sanitization,
    and ultra-low latency early first-word emission.
    """

    def __init__(self, *args, first_chunk_words: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_chunk_words = first_chunk_words
        if first_chunk_words > 0:
            self._text_aggregator = EarlyWordAggregator(first_words_count=first_chunk_words)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, LLMFullResponseStartFrame):
            if hasattr(self._text_aggregator, "reset"):
                await self._text_aggregator.reset()
        await super().process_frame(frame, direction)

    async def run_tts(self, text: str, context_id: str):
        if not text or not text.strip():
            return
        # Clean unicode characters that cause issues with TTS/codecs
        sanitized = (
            text.replace("\u2011", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u202f", " ")
            .replace("\u00a0", " ")
            .replace("\u2019", "'")
            .replace("\u2018", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2026", "...")
            .replace("$", " dollars ")
        )
        async for frame in super().run_tts(sanitized, context_id):
            if isinstance(frame, TTSAudioRawFrame):
                frame.sample_rate = 22050
            yield frame


_GLOBAL_SARVAM_SESSION: Optional[aiohttp.ClientSession] = None


def get_sarvam_session() -> aiohttp.ClientSession:
    """Persistent pooled aiohttp session for Sarvam AI TTS to eliminate connection cold start."""
    global _GLOBAL_SARVAM_SESSION
    if _GLOBAL_SARVAM_SESSION is None or _GLOBAL_SARVAM_SESSION.closed:
        connector = aiohttp.TCPConnector(keepalive_timeout=120, limit=20, force_close=False)
        _GLOBAL_SARVAM_SESSION = aiohttp.ClientSession(connector=connector)
    return _GLOBAL_SARVAM_SESSION


async def close_sarvam_session():
    global _GLOBAL_SARVAM_SESSION
    if _GLOBAL_SARVAM_SESSION and not _GLOBAL_SARVAM_SESSION.closed:
        await _GLOBAL_SARVAM_SESSION.close()
        _GLOBAL_SARVAM_SESSION = None


async def run_bot(transport: BaseTransport, handle_sigint: bool, call_id: str = None):
    load_dotenv(override=True)
    # ── API Keys ──
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")

    sarvam_key = os.getenv("SARVAM_API_KEY")

    # ── STT Service Setup (Shunya Labs or Deepgram) ──
    stt_provider = os.getenv("STT_PROVIDER", "deepgram").lower().strip()

    if stt_provider in ("shunyalabs", "shunya"):
        from pipecat_shunyalabs import ShunyalabsSTTService

        shunyalabs_key = os.getenv("SHUNYALABS_API_KEY")
        shunyalabs_lang = os.getenv("SHUNYALABS_LANGUAGE", "en")
        shunyalabs_endpoint_silence = int(os.getenv("SHUNYALABS_ENDPOINT_SILENCE_MS", "350"))
        shunyalabs_vad = os.getenv("SHUNYALABS_VAD", "silero")
        shunyalabs_model = os.getenv("SHUNYALABS_MODEL", "").strip() or None
        shunyalabs_codeswitch = os.getenv("SHUNYALABS_CODESWITCH", "false").lower() == "true"

        stt = ShunyalabsSTTService(
            api_key=shunyalabs_key,
            language=shunyalabs_lang,
            sample_rate=8000,
            min_send_bytes=1600,  # 100ms at 8kHz 16-bit mono PCM
            endpoint_silence_ms=shunyalabs_endpoint_silence,
            vad=shunyalabs_vad,
            model=shunyalabs_model,
            codeswitch=shunyalabs_codeswitch,
        )
        logger.info(
            f"STT: Shunya Labs ({shunyalabs_model or 'default'}, {shunyalabs_lang}, "
            f"endpoint_silence={shunyalabs_endpoint_silence}ms, vad={shunyalabs_vad}, codeswitch={shunyalabs_codeswitch})"
        )
    else:
        from pipecat.services.deepgram.stt import DeepgramSTTService

        deepgram_model = os.getenv("DEEPGRAM_MODEL", "nova-3-general")
        deepgram_lang = os.getenv("DEEPGRAM_LANGUAGE", "en")
        deepgram_endpointing = int(os.getenv("DEEPGRAM_ENDPOINTING", "350"))

        stt = DeepgramSTTService(
            api_key=deepgram_key,
            sample_rate=8000,
            encoding="linear16",
            settings=DeepgramSTTService.Settings(
                model=deepgram_model,
                language=deepgram_lang,
                smart_format=True,
                punctuate=True,
                endpointing=deepgram_endpointing,
                keyterm=[
                    # Brand & Suite
                    "Nimbus", "Nimbus suite", "suite",
                    # All 24 Nimbus Products (Full names & standalone terms)
                    "Nimbus DataPipe", "DataPipe", "data pipe",
                    "Nimbus Vault", "Vault", "password manager",
                    "Nimbus SSO", "SSO", "single sign on",
                    "Nimbus Endpoint", "Endpoint",
                    "Nimbus CRM", "CRM", "Leads", "Nimbus Leads",
                    "Nimbus Quote", "Quote", "Campaigns", "Nimbus Campaigns",
                    "Nimbus Social", "Social", "Nimbus Sites", "Sites",
                    "Nimbus Books", "Books", "Nimbus Invoice", "Invoice", "Expense", "Nimbus Expense",
                    "Nimbus People", "People", "Nimbus Recruit", "Recruit", "Nimbus Payroll", "Payroll",
                    "Nimbus Desk", "Desk", "Nimbus Chat", "Chat", "Nimbus Knowledge", "Knowledge",
                    "Nimbus Projects", "Projects", "Nimbus Docs", "Docs", "Nimbus Boards", "Boards",
                    "Nimbus Analytics", "Analytics", "Nimbus Dashboards", "Dashboards",
                    # Business, Pricing & Policy Terms
                    "pricing", "price", "plan", "starter", "professional", "enterprise",
                    "free trial", "refund", "cancel", "cancellation", "billing",
                    "support", "subscription", "discount", "annual", "monthly",
                    "cheapest", "cheapest products", "most affordable", "affordable",
                    "expensive", "most expensive", "expensive product", "costly", "budget",
                    "compare", "comparison", "lowest", "highest",
                    "Vobiz", "Prathamesh", "help", "question", "questions",
                ],
            ),
        )
        logger.info(f"STT: Deepgram Nova-3 ({deepgram_model}, {deepgram_lang}, endpointing={deepgram_endpointing}ms)")

    # ── LLM: Groq (Ultra-Low Latency & High Reasoning) ──
    from pipecat.services.groq.llm import GroqLLMService

    groq_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    groq_temp = float(os.getenv("GROQ_TEMPERATURE", "0.3"))
    groq_max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "150"))
    llm = GroqLLMService(
        api_key=groq_key,
        settings=GroqLLMService.Settings(
            model=groq_model,
            temperature=groq_temp,
            max_tokens=groq_max_tokens,
        ),
    )
    logger.info(f"LLM: Groq ({groq_model}, temp={groq_temp}, max_tokens={groq_max_tokens})")

    # ── TTS: Sarvam AI (Crystal Clear Native 22050Hz -> 8000Hz Telephony Resampling) ──
    sarvam_session = get_sarvam_session()
    sarvam_model = os.getenv("SARVAM_MODEL", "bulbul:v3")
    sarvam_voice = os.getenv("SARVAM_VOICE", "priya")
    sarvam_lang = os.getenv("SARVAM_LANGUAGE", "en-IN")
    tts_first_chunk_words = int(os.getenv("TTS_FIRST_CHUNK_WORDS", "0"))

    tts = FixedSarvamHttpTTSService(
        api_key=sarvam_key,
        aiohttp_session=sarvam_session,
        settings=SarvamHttpTTSService.Settings(
            model=sarvam_model,
            voice=sarvam_voice,
            language=sarvam_lang,
            pace=1.05,
            enable_preprocessing=True,
        ),
        sample_rate=8000,
        first_chunk_words=tts_first_chunk_words,
    )
    tts._sample_rate = 8000
    logger.info(f"TTS: Sarvam AI ({sarvam_model}, voice: {sarvam_voice}, {sarvam_lang}, early_first_chunk_words={tts_first_chunk_words})")

    # ── Hangup Processor & end_call Tool ──
    hangup_processor = InterruptionAwareHangupProcessor(hangup_delay=5.0)

    async def end_call_handler(params: FunctionCallParams, *args, **kwargs):
        logger.info("📞 [Tool] 'end_call' triggered by assistant. Arming interruption-aware hangup.")
        hangup_processor.trigger_hangup()
        result_cb = getattr(params, "result_callback", None)
        if callable(result_cb):
            await result_cb({"status": "call_ending", "message": "Signoff initiated. Call will end after goodbye."})
        elif len(args) >= 5 and callable(args[4]):
            await args[4]({"status": "call_ending", "message": "Signoff initiated. Call will end after goodbye."})

    end_call_schema = FunctionSchema(
        name="end_call",
        description=(
            "Call this function ONLY when the user explicitly indicates they are finished and want to end the call "
            "(e.g., 'no', 'that is all', 'bye', 'thank you bye', 'all good', 'no more questions'). "
            "NEVER call this function if the user is asking questions or says 'wait' / 'one more thing'."
        ),
        properties={},
        required=[],
        handler=end_call_handler,
    )

    # ── System Instructions (Simple RAG-powered) ──
    greeting_text = "Hello! Welcome to Nimbus. How can I help you today?"

    system_prompt = (
        "You are a friendly, professional customer support voice agent for Nimbus, "
        "a cloud software company headquartered in Austin, Texas. You are speaking on a live phone call.\n\n"
        "VOICE SPEED & LATENCY RULES:\n"
        "- DIRECT SPEECH: You are speaking on a live telephone call. Begin speaking your answer immediately. Do not produce preamble, thought process, or meta-commentary.\n"
        "- FAST FIRST SENTENCE: Always start your response with a short, punchy first sentence under 10 words "
        "(e.g., 'Nimbus Docs is our real-time document collaboration tool.'). "
        "This allows the voice synthesizer to start speaking immediately with zero lag.\n"
        "- CONCISENESS: Keep the entire answer between 1 and 3 short sentences total. The caller is on a phone, not reading a screen.\n"
        "- NO MARKDOWN & PLAIN ASCII: Never use markdown formatting, bullet points, asterisks, citations, or emojis. Use standard words and basic punctuation only (no special unicode hyphens or symbols).\n"
        "- NATURAL PRICING: Speak prices naturally and avoid repetitive filler phrases. "
        "Say 'fifteen dollars per user per month' rather than '$15/user/mo'. "
        "For multi-product comparisons, say 'Nimbus Docs at eight dollars, plus Nimbus People and Nimbus Invoice at ten dollars a month.'\n"
        "- ENTERPRISE: For Enterprise tiers, state that pricing is custom and invite them to contact our sales team.\n"
        "- ACCURACY & RAG: You are equipped with real-time Simple RAG. Relevant product features, pricing tiers, "
        "and policies are supplied in your prompt under '[Retrieved Website Knowledge]'. Always rely on this knowledge.\n"
        "- COMPARISONS: When answering comparative questions (cheapest products, most expensive, price ranges), "
        "name the top products concisely.\n"
        "- CALL SIGNOFF: When the user indicates they are finished or saying goodbye "
        "(e.g., 'no', 'that is all', 'bye', 'thank you', 'no more questions'), "
        "give a warm Nimbus signoff ('Thank you for calling Nimbus. Have a wonderful day! Goodbye.') "
        "and immediately call the 'end_call' function.\n\n"
        "COMPANY INFORMATION & POLICIES:\n"
        f"{NIMBUS_COMPANY_INFO}\n\n"
        "NIMBUS 24-PRODUCT SUITE (BY CATEGORY):\n"
        "- Sales & Marketing: Nimbus CRM, Nimbus Leads, Nimbus Quote, Nimbus Campaigns, Nimbus Social, Nimbus Sites\n"
        "- Finance: Nimbus Books, Nimbus Invoice, Nimbus Expense\n"
        "- Human Resources: Nimbus People, Nimbus Recruit, Nimbus Payroll\n"
        "- Customer Support: Nimbus Desk, Nimbus Chat, Nimbus Knowledge\n"
        "- Productivity & Collaboration: Nimbus Projects, Nimbus Docs, Nimbus Boards\n"
        "- Analytics & BI: Nimbus Analytics, Nimbus Dashboards, Nimbus DataPipe (no-code ETL & automated data integration pipelines)\n"
        "- IT & Security: Nimbus Vault (secrets & passwords), Nimbus SSO (single sign-on & identity), Nimbus Endpoint (device management)\n\n"
        "- Universal Guarantee: All products include a Free tier ($0), a 14-day free trial on paid plans (no credit card required), "
        "and a 30-day money-back guarantee.\n"
        "- Annual Discount: 20% discount when billed annually.\n"
        "- Data Residency: Regional data centers available in the US, European Union (Frankfurt), India, and Australia.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": greeting_text},
    ]

    # ── Context + Robust Silero VAD (Telephony-Tuned for Volume Alterations) ──
    context = LLMContext(messages, tools=[end_call_schema])
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.35,      # Catches soft, quiet, and low-volume speaking
            start_secs=0.20,      # Faster onset detection for natural barge-in
            stop_secs=0.20,       # Prevents cutting caller off mid-sentence during brief pauses
            min_volume=0.03,     # Low threshold accommodates wide dynamic range from whisper to loud
        )
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad_analyzer),
    )

    from observability.collector import CallTelemetryCollector

    collector = CallTelemetryCollector(call_id=call_id or "unknown_call", context=context)

    # ── Semantic Response Cache & RAG Processor ──
    rag_processor = SemanticRAGProcessor(rag, semantic_cache, hangup_processor=hangup_processor)

    # ── Pipeline ──
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            rag_processor,
            llm,
            collector,
            tts,
            transport.output(),
            context_aggregator.assistant(),
            hangup_processor,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    hangup_processor.set_task(task)

    # ── Greeting: played immediately when call connects ──
    async def play_greeting():
        # Short 150ms buffer to allow WebSocket start frame negotiation to complete cleanly
        await asyncio.sleep(0.15)
        if _GREETING_RAW_AUDIO:
            logger.info(f"📞 Playing pre-recorded greeting immediately ({len(_GREETING_RAW_AUDIO)} bytes, 8000Hz)...")
            try:
                await task.queue_frame(TTSStartedFrame())
                await task.queue_frame(
                    TTSAudioRawFrame(
                        audio=_GREETING_RAW_AUDIO,
                        sample_rate=8000,
                        num_channels=1,
                    )
                )
                await task.queue_frame(TTSStoppedFrame())
            except Exception as e:
                logger.warning(f"Failed to play pre-recorded greeting: {e}")
        else:
            # Fallback to TTS speak frame if audio file was missing
            logger.info(f"📞 Playing greeting via fallback TTS: '{greeting_text}'")
            try:
                await task.queue_frame(TTSSpeakFrame(greeting_text))
            except Exception as e:
                logger.warning(f"Failed to queue greeting: {e}")

    # ── Disconnect handler ──
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Call ended (client disconnected)")
        collector.finalize()
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    greeting_task = asyncio.create_task(play_greeting())

    try:
        await runner.run(task)
    finally:
        collector.finalize()
        if not greeting_task.done():
            greeting_task.cancel()


async def bot(runner_args: RunnerArguments, call_id: str = None, stream_id: str = None):
    """Main bot entry point compatible with Pipecat Cloud."""

    env_encoding = os.getenv("VOBIZ_ENCODING", "audio/x-mulaw")
    env_sample_rate = int(os.getenv("VOBIZ_SAMPLE_RATE", "8000"))

    try:
        parsed = await parse_vobiz_start(runner_args.websocket)
    except WebSocketDisconnect:
        logger.warning("Vobiz WebSocket disconnected before start event.")
        return

    logger.info(
        f"Vobiz start: callId={parsed['call_id']!r}, streamId={parsed['stream_id']!r}, "
        f"mediaFormat=({parsed['encoding']!r}, {parsed['sample_rate']})"
    )
    call_id = call_id or parsed["call_id"]
    stream_id = stream_id or parsed["stream_id"]
    vobiz_encoding = parsed["encoding"] or env_encoding
    vobiz_sample_rate = parsed["sample_rate"] or env_sample_rate

    serializer = FixedVobizFrameSerializer(
        stream_id=stream_id,
        call_id=call_id,
        auth_id=os.getenv("VOBIZ_AUTH_ID", ""),
        auth_token=os.getenv("VOBIZ_AUTH_TOKEN", ""),
        params=VobizFrameSerializer.InputParams(
            vobiz_sample_rate=vobiz_sample_rate,
            encoding=vobiz_encoding,
            sample_rate=None,
            l16_byte_order=os.getenv("VOBIZ_L16_ENDIAN", "be"),
            auto_hang_up=True,
            hangup_method=HANGUP_BOTH,
        ),
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    await run_bot(transport, runner_args.handle_sigint, call_id=call_id)
