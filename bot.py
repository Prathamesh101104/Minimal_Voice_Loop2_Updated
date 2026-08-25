#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

import aiohttp
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
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
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

load_dotenv(override=True)


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


class FixedSarvamHttpTTSService(SarvamHttpTTSService):
    """Sarvam AI HTTP TTS with proper native sample rate labelling.

    Sarvam returns audio at 22050 Hz regardless of requested sample_rate.
    This class ensures TTSAudioRawFrame.sample_rate is correctly set to 22050
    so that downstream resamplers (like Vobiz 8kHz mu-law converter) properly
    resample the audio down to 8000 Hz for crystal clear, natural human speech.
    """

    async def run_tts(self, text: str, context_id: str):
        async for frame in super().run_tts(text, context_id):
            if isinstance(frame, TTSAudioRawFrame):
                frame.sample_rate = 22050
            yield frame


async def run_bot(transport: BaseTransport, handle_sigint: bool, call_id: str = None):
    # ── API Keys ──
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    sarvam_key = os.getenv("SARVAM_API_KEY")

    # ── STT: Deepgram Nova-3 Streaming (Low Latency 8kHz) ──
    from pipecat.services.deepgram.stt import DeepgramSTTService

    deepgram_model = os.getenv("DEEPGRAM_MODEL", "nova-3-general")
    deepgram_lang = os.getenv("DEEPGRAM_LANGUAGE", "en")
    deepgram_endpointing = int(os.getenv("DEEPGRAM_ENDPOINTING", "120"))

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
                "richest", "state", "states", "India", "GDP", "fastest", "animal",
                "cheetah", "Prathamesh", "Vobiz", "Maharashtra", "Bengaluru", "Delhi",
                "capital", "currency", "weather", "help", "question", "questions",
                "country", "countries", "population", "president", "minister"
            ],
        ),
    )
    logger.info(f"STT: Deepgram Nova-3 ({deepgram_model}, {deepgram_lang}, endpointing={deepgram_endpointing}ms)")

    # ── LLM: Groq (Ultra-Low Latency) ──
    from pipecat.services.groq.llm import GroqLLMService

    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    llm = GroqLLMService(
        api_key=groq_key,
        settings=GroqLLMService.Settings(
            model=groq_model,
            temperature=0.6,
            max_tokens=200,
        ),
    )
    logger.info(f"LLM: Groq ({groq_model})")

    # ── TTS: Sarvam AI (Crystal Clear Native 22050Hz -> 8000Hz Telephony Resampling) ──
    sarvam_session = aiohttp.ClientSession()
    sarvam_model = os.getenv("SARVAM_MODEL", "bulbul:v3")
    sarvam_voice = os.getenv("SARVAM_VOICE", "priya")
    sarvam_lang = os.getenv("SARVAM_LANGUAGE", "en-IN")

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
    )
    tts._sample_rate = 8000
    logger.info(f"TTS: Sarvam AI ({sarvam_model}, voice: {sarvam_voice}, {sarvam_lang})")

    # ── Hangup Processor & end_call Tool ──
    hangup_processor = InterruptionAwareHangupProcessor(hangup_delay=4.0)

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

    # ── System Instructions ──
    greeting_text = "Hello! How can I help you today?"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly, fast, and helpful voice assistant on a live phone call. "
                "Keep every answer very concise (1-2 sentences maximum). "
                "Never use markdown formatting, bullet points, asterisks, or emojis. "
                "When the user indicates they have no more questions or are saying goodbye (e.g. 'no', 'that's all', 'bye', 'thank you', 'no more questions'), "
                "give a warm signoff (e.g. 'Thank you for calling. Have a great day! Goodbye.') and call the 'end_call' function."
            ),
        },
        {
            "role": "assistant",
            "content": greeting_text,
        },
    ]

    # ── Context + Robust Silero VAD ──
    context = LLMContext(messages, tools=[end_call_schema])
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.65,
            start_secs=0.20,
            stop_secs=0.25,
            min_volume=0.5,
        )
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad_analyzer),
    )

    # ── Pipeline ──
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
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

    # ── Greeting: dispatched safely after pipeline starts ──
    async def play_greeting():
        await asyncio.sleep(1.2)
        logger.info(f"📞 Playing greeting: '{greeting_text}' (tts.sample_rate={tts.sample_rate})")
        try:
            await task.queue_frame(TTSSpeakFrame(greeting_text))
        except Exception as e:
            logger.warning(f"Failed to queue greeting: {e}")

    # ── Disconnect handler ──
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Call ended (client disconnected)")
        try:
            if sarvam_session and not sarvam_session.closed:
                await sarvam_session.close()
        except Exception:
            pass
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    greeting_task = asyncio.create_task(play_greeting())

    try:
        await runner.run(task)
    finally:
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
